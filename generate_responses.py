import pandas as pd
import glob
import os
import random
import argparse
import asyncio
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# ---------------------------------------------------------------------------
# Model / pipeline initialisation (singleton per process)
# ---------------------------------------------------------------------------

_hf_pipeline = None
_hf_model_name = None


def get_hf_pipeline(model_name: str, max_new_tokens: int = 2048):
    """
    Lazily initialise (and cache) a HuggingFace text-generation pipeline.
    Loads the model once and reuses it for every call.
    """
    global _hf_pipeline, _hf_model_name
    if _hf_pipeline is None or _hf_model_name != model_name:
        print(f"Loading HuggingFace model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        _hf_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )
        _hf_model_name = model_name
        print(f"Model loaded on device(s): {model.hf_device_map if hasattr(model, 'hf_device_map') else 'auto'}")
    return _hf_pipeline


# ---------------------------------------------------------------------------
# Core generation helpers
# ---------------------------------------------------------------------------

def _generate_one(model_name: str, messages: list, max_tokens: int, temperature: float) -> str:
    """
    Synchronous single-sample generation via HuggingFace pipeline.
    Converts the OpenAI-style messages list using the model's chat template.
    """
    pipe = get_hf_pipeline(model_name, max_new_tokens=max_tokens)
    tokenizer = pipe.tokenizer

    # apply_chat_template handles system / user / assistant roles
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    outputs = pipe(
        prompt_text,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        top_p=0.95,
        return_full_text=False,   # return only newly generated tokens
    )
    return outputs[0]["generated_text"]


async def generate_responses_async(
    model_name,          # replaces `client` — we pass the name directly
    formatted_prompts,
    max_tokens,
    temperature,
    semaphore,
    K=1,
):
    """
    Generates responses from a HuggingFace model for a batch of prompts.

    Args:
        model_name: HuggingFace model ID, e.g. 'meta-llama/Llama-3.1-8B-Instruct'
        formatted_prompts: Dict mapping prompt keys to OpenAI-style message lists
        max_tokens: Maximum new tokens to generate
        temperature: Sampling temperature
        semaphore: asyncio.Semaphore for concurrency limiting
        K: Number of independent responses per prompt (default=1)

    Returns:
        dict: Mapping of prompt keys to lists of generated responses
    """
    results = {}

    async def process_prompt(prompt_key, messages):
        retry_delay = 3
        max_retries = 7
        jitter = 0.1
        all_responses = []

        for _ in range(K):
            for attempt in range(max_retries):
                try:
                    async with semaphore:
                        # Run blocking HF inference in a thread pool
                        response = await asyncio.to_thread(
                            _generate_one,
                            model_name,
                            messages,
                            max_tokens,
                            temperature,
                        )
                    all_responses.append(response)
                    break  # success — move on

                except Exception as e:
                    if attempt == max_retries - 1:
                        print(f"Max retries exceeded for prompt: {prompt_key}")
                        print(e)
                        all_responses.append(None)
                    else:
                        print(
                            f"Error for prompt: {prompt_key}. "
                            f"Attempt {attempt + 1}/{max_retries}. "
                            f"Retrying in {retry_delay:.1f}s."
                        )
                        print(e)
                        jittered_delay = retry_delay * (
                            1 + jitter * (2 * random.random() - 1)
                        )
                        await asyncio.sleep(jittered_delay)
                        retry_delay *= 2

        results[prompt_key] = all_responses

    tasks = [process_prompt(key, messages) for key, messages in formatted_prompts.items()]
    await asyncio.gather(*tasks)
    return results


def create_model_client(model_name):
    """
    For HuggingFace models we don't need a separate client object.
    We just return the model_name string so call-sites stay compatible.
    """
    # Eagerly load the pipeline at startup so the first request isn't slow.
    get_hf_pipeline(model_name)
    return model_name   # used as `client` throughout process_dataframe


# ---------------------------------------------------------------------------
# DataFrame processing (unchanged logic, only signature adapted)
# ---------------------------------------------------------------------------

async def process_dataframe(
    input_csv,
    output_csv,
    temperature,
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    max_tokens=2000,
    belief_k=None,
    lie_k=10,
):
    """
    Process a dataframe and generate responses from a HuggingFace model.

    Args:
        input_csv (str): Path to input CSV file
        output_csv (str): Path to output CSV file
        temperature (float): Sampling temperature for generation
        model_name (str): HuggingFace model ID
        max_tokens (int): Maximum new tokens for generation
        belief_k (dict): Mapping of belief-elicitation indices to run counts
                         e.g. {"1": 3, "2": 1, "3": 1}
        lie_k (int): Number of lying-prompt runs per row
    """
    # `client` is just the model name for HF
    client = create_model_client(model_name)

    # Limit concurrent threads — tune to your GPU count / VRAM
    semaphore = asyncio.Semaphore(1)

    df = pd.read_csv(input_csv)
    if "belief_system_1" not in df.columns and "belief_elicit_1" in df.columns:
        df["belief_system_1"] = "You are a helpful assistant."

    is_doubling_down = (
        "dd_turn_1_prompt" in df.columns and "dd_turn_1_response" in df.columns
    )
    print(f"Is this a doubling down scenario? {is_doubling_down}")

    all_prompts = []
    indices = []
    prompt_types = []

    for index, row in df.iterrows():
        # --- lying / doubling-down prompts ---
        if is_doubling_down:
            for run in range(lie_k):
                all_prompts.append([
                    {"role": "system",    "content": row["system_prompt"]},
                    {"role": "user",      "content": row["dd_turn_1_prompt"]},
                    {"role": "assistant", "content": row["dd_turn_1_response"]},
                    {"role": "user",      "content": row["user_prompt"]},
                ])
                indices.append(index)
                prompt_types.append(f"lying_run{run+1}")
        else:
            for run in range(lie_k):
                all_prompts.append([
                    {"role": "system", "content": row["system_prompt"]},
                    {"role": "user",   "content": row["user_prompt"]},
                ])
                indices.append(index)
                prompt_types.append(f"lying_run{run+1}")

        # --- belief-elicitation prompts ---
        if "belief_elicit_1" in df.columns:
            for be_num, k_value in belief_k.items():
                belief_col = f"belief_elicit_{be_num}"
                if pd.notna(row.get(belief_col)):
                    for run in range(k_value):
                        all_prompts.append([
                            {"role": "system", "content": row["belief_system_1"]},
                            {"role": "user",   "content": row[belief_col]},
                        ])
                        indices.append(index)
                        prompt_types.append(f"be{be_num}_run{run+1}")

    # --- generate all prompts ---
    # NOTE: `client` == model_name for HuggingFace
    all_responses = await asyncio.gather(*[
        generate_responses_async(
            client,
            {prompt_type: prompt},
            max_tokens=max_tokens,
            temperature=temperature,
            semaphore=semaphore,
            K=1,
        )
        for prompt, prompt_type in zip(all_prompts, prompt_types)
    ])

    # --- column name mapping ---
    column_mapping = {
        f"lying_run{run+1}": f"generation(System Prompt + User Prompt)_run{run+1}"
        for run in range(lie_k)
    }
    if "belief_elicit_1" in df.columns:
        for be_num, k_value in belief_k.items():
            for run in range(k_value):
                column_mapping[f"be{be_num}_run{run+1}"] = (
                    f"generation(belief_system_1 + belief_elicit_{be_num})_run{run+1}"
                )

    # --- write responses back to dataframe ---
    for response, index, prompt_type in zip(all_responses, indices, prompt_types):
        column_name = column_mapping[prompt_type]
        try:
            response_value = response[prompt_type][0] if prompt_type in response else None
            if response_value is None:
                raise ValueError("Generation failed or timed out")
            df.at[index, column_name] = response_value
        except Exception as e:
            error_msg = f"[ERROR: {str(e)}]"
            print(f"WARNING: Error processing response for index {index}, "
                  f"prompt type {prompt_type}: {str(e)}")
            df.at[index, column_name] = error_msg

    df.to_csv(output_csv, index=False)
    print(f"Saved results to {output_csv}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate responses from a HuggingFace model for a set of prompts"
    )
    parser.add_argument("--test",        action="store_true", help="Run in test mode")
    parser.add_argument("--model",       default="meta-llama/Llama-3.1-8B-Instruct",
                        help="HuggingFace model ID")
    parser.add_argument("--max_tokens",  type=int,   default=1000,
                        help="Maximum new tokens for generation")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for generation")
    parser.add_argument("--lie_k",       type=int,   default=1,
                        help="Number of lying prompts to generate for each row")
    args = parser.parse_args()

    pattern = "test_csv_data/*.csv" if args.test else "csv_data/*.csv"
    for file in glob.iglob(pattern):
        belief_k = {"1": 3} if "statistics" in file else {"1": 3, "2": 1, "3": 1}

        # Use the last path component of the model ID as the filename suffix
        modelname = args.model.split("/")[-1] if "/" in args.model else args.model
        output_file = file.replace(".csv", f"_{modelname}.csv")
        output_file = output_file.replace("csv_data/", "csv_data/responses/")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        if os.path.exists(output_file):
            print(f"Skipping {output_file} as it already exists")
            continue

        print("Processing", file, "->", output_file)
        asyncio.run(process_dataframe(
            file,
            output_file,
            temperature=args.temperature,
            model_name=args.model,
            max_tokens=args.max_tokens,
            belief_k=belief_k,
            lie_k=args.lie_k,
        ))
