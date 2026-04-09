import pandas as pd
import glob
import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from peft import PeftModel
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Model / tokenizer initialisation (singleton per process)
# ---------------------------------------------------------------------------

_model = None
_tokenizer = None
_hf_model_name = None


def get_model_and_tokenizer(model_name: str, adapter_path: str | None):
    """
    Lazily initialise (and cache) the model and tokenizer directly.
    Uses model.generate() instead of HF pipeline to enable true batching
    across multiple GPUs. device_map="auto" is fine for *loading* the model
    across GPUs — the issue was the pipeline wrapper, not device_map itself.
    """
    global _model, _tokenizer, _hf_model_name
    if _model is None or _hf_model_name != model_name:
        print(f"Loading base model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",  # critical on H100 — ~2x attention speed
        )

        if adapter_path is not None:
            print(f"Loading adapter from: {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path)
            model = model.merge_and_unload()

        model.eval()

        _model = model
        _tokenizer = tokenizer
        _hf_model_name = model_name
        print(
            f"Model loaded on device(s): "
            f"{model.hf_device_map if hasattr(model, 'hf_device_map') else 'auto'}"
        )
    return _model, _tokenizer


# ---------------------------------------------------------------------------
# Core generation — true batching via model.generate()
# ---------------------------------------------------------------------------

def generate_batch(
    model_name: str,
    adapter_path: str | None,
    prompt_texts: list[str],
    max_tokens: int,
    temperature: float,
    batch_size: int = 32,       # H100: 32-64 for 70B, 64-128 for 8B
    preview_chars: int = 120,
) -> list[str]:
    """
    Run inference using model.generate() directly — bypasses the HF pipeline
    wrapper which was silently preventing true batched GPU execution.

    Key fixes vs. original:
    - No pipeline wrapper: model.generate() does a single batched forward pass
      per batch, not N sequential ones.
    - Tokenization is explicit and batched — padding/truncation happens once
      per batch, not per sample.
    - Input tokens are stripped from output manually (more reliable than
      return_full_text=False in the pipeline).
    - torch.no_grad() + torch.cuda.empty_cache() keeps memory clean across
      large batch runs.

    Args:
        prompt_texts:  List of strings (chat template already applied).
        batch_size:    Number of prompts per GPU batch.
                       Recommended: 8B→64-128, 70B on 4xH100→16-32, 70B on 8xH100→32-64
        preview_chars: How many characters of each response to print inline.

    Returns:
        List of generated strings, one per input prompt, in the same order.
    """
    model, tokenizer = get_model_and_tokenizer(model_name, adapter_path)
    all_responses: list[str] = []

    batches = [prompt_texts[i:i + batch_size] for i in range(0, len(prompt_texts), batch_size)]

    with tqdm(total=len(prompt_texts), unit="prompt", desc="Generating") as pbar:
        for batch in batches:
            # Tokenize the whole batch at once — padding/truncation in one shot
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to("cuda:0")  # first GPU; device_map handles the rest of the model

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    top_p=0.95,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True,
                )

            # Strip input tokens — only decode the newly generated tokens
            input_len = inputs["input_ids"].shape[1]
            for seq in outputs:
                text = tokenizer.decode(seq[input_len:], skip_special_tokens=True)
                all_responses.append(text)
                preview = text.replace("\n", " ").strip()[:preview_chars]
                tqdm.write(f"  → {preview}")

            pbar.update(len(batch))
            torch.cuda.empty_cache()  # prevents fragmentation across large runs

    return all_responses


# ---------------------------------------------------------------------------
# DataFrame processing
# ---------------------------------------------------------------------------

def process_dataframe(
    input_csv: str,
    output_csv: str,
    temperature: float,
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    adapter_path: str | None = None,
    max_tokens: int = 512,
    belief_k: dict | None = None,
    lie_k: int = 1,
    batch_size: int = 32,
) -> None:
    """
    Process a dataframe and generate responses from a HuggingFace model.
    """
    df = pd.read_csv(input_csv)

    if "belief_system_1" not in df.columns and "belief_elicit_1" in df.columns:
        df["belief_system_1"] = "You are a helpful assistant."

    is_doubling_down = (
        "dd_turn_1_prompt" in df.columns and "dd_turn_1_response" in df.columns
    )
    print(f"Is this a doubling down scenario? {is_doubling_down}")

    # ------------------------------------------------------------------
    # 1. Collect every (row_index, prompt_type, messages) triple up front
    # ------------------------------------------------------------------
    records: list[tuple[int, str, list[dict]]] = []

    for index, row in df.iterrows():
        if is_doubling_down:
            messages = [
                {"role": "system",    "content": row["system_prompt"]},
                {"role": "user",      "content": row["dd_turn_1_prompt"]},
                {"role": "assistant", "content": row["dd_turn_1_response"]},
                {"role": "user",      "content": row["user_prompt"]},
            ]
        else:
            messages = [
                {"role": "system", "content": row["system_prompt"]},
                {"role": "user",   "content": row["user_prompt"]},
            ]

        for run in range(lie_k):
            records.append((index, f"lying_run{run + 1}", messages))

        if "belief_elicit_1" in df.columns:
            for be_num, k_value in (belief_k or {}).items():
                belief_col = f"belief_elicit_{be_num}"
                if pd.notna(row.get(belief_col)):
                    be_messages = [
                        {"role": "system", "content": row["belief_system_1"]},
                        {"role": "user",   "content": row[belief_col]},
                    ]
                    for run in range(k_value):
                        records.append((index, f"be{be_num}_run{run + 1}", be_messages))

    # ------------------------------------------------------------------
    # 2. Apply chat template to every prompt (vectorised, once)
    #    Load tokenizer directly — no pipeline needed here anymore
    # ------------------------------------------------------------------
    _, tokenizer = get_model_and_tokenizer(model_name, adapter_path)

    prompt_texts = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        for _, _, messages in records
    ]

    # ------------------------------------------------------------------
    # 3. Single batched GPU call
    # ------------------------------------------------------------------
    print(f"Generating {len(prompt_texts)} responses (batch_size={batch_size})...")
    responses = generate_batch(
        model_name=model_name,
        adapter_path=adapter_path,
        prompt_texts=prompt_texts,
        max_tokens=max_tokens,
        temperature=temperature,
        batch_size=batch_size,
    )

    # ------------------------------------------------------------------
    # 4. Column name mapping
    # ------------------------------------------------------------------
    column_mapping = {
        f"lying_run{run + 1}": f"generation(System Prompt + User Prompt)_run{run + 1}"
        for run in range(lie_k)
    }
    if "belief_elicit_1" in df.columns:
        for be_num, k_value in (belief_k or {}).items():
            for run in range(k_value):
                column_mapping[f"be{be_num}_run{run + 1}"] = (
                    f"generation(belief_system_1 + belief_elicit_{be_num})_run{run + 1}"
                )

    # ------------------------------------------------------------------
    # 5. Write results back to dataframe
    # ------------------------------------------------------------------
    for (df_index, prompt_type, _), response in zip(records, responses):
        col = column_mapping[prompt_type]
        if response is None:
            df.at[df_index, col] = "[ERROR: generation returned None]"
        else:
            df.at[df_index, col] = response

    df.to_csv(output_csv, index=False)
    print(f"Saved results to {output_csv}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate responses from a HuggingFace model for a set of prompts"
    )
    parser.add_argument("--test",         action="store_true", help="Run in test mode")
    parser.add_argument("--lie",          action="store_true", help="Run in lie mode (no belief elicitation)")
    parser.add_argument("--model",        default="meta-llama/Llama-3.1-8B-Instruct",
                        help="HuggingFace model ID")
    parser.add_argument("--max_tokens",   type=int,   default=1000,
                        help="Maximum new tokens for generation")
    parser.add_argument("--temperature",  type=float, default=1.0,
                        help="Temperature for generation")
    parser.add_argument("--lie_k",        type=int,   default=1,
                        help="Number of response samples per lying prompt per row")
    parser.add_argument("--batch_size",   type=int,   default=32,
                        help="GPU batch size — 8B: try 64-128, 70B 4xH100: try 16-32, 70B 8xH100: try 32-64")
    parser.add_argument("--adapter_path", default=None,
                        help="Local path to PEFT adapter folder")
    args = parser.parse_args()

    if args.test:
        pattern = "test_csv_data/*.csv"
    elif args.lie:
        pattern = "lie_csv_data/*.csv"
    else:
        pattern = "csv_data/*.csv"

    for file in glob.iglob(pattern):
        belief_k = {"1": 3} if "statistics" in file else {"1": 3, "2": 1, "3": 1}

        modelname = args.model.split("/")[-1] if "/" in args.model else args.model
        adapterstyle = args.adapter_path.split("/")[0] if "/" in args.adapter_path else args.adapter_path
        adaptername = args.adapter_path.split("/")[1] if "/" in args.adapter_path else args.adapter_path
        output_file = file.replace(".csv", f"_{modelname}_{adapterstyle}{adaptername}.csv")
        output_file = output_file.replace("csv_data/", "csv_data/responses/")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        if os.path.exists(output_file):
            print(f"Skipping {output_file} as it already exists")
            continue

        print("Processing", file, "->", output_file)
        process_dataframe(
            file,
            output_file,
            temperature=args.temperature,
            model_name=args.model,
            max_tokens=args.max_tokens,
            belief_k=belief_k,
            lie_k=args.lie_k,
            batch_size=args.batch_size,
            adapter_path=args.adapter_path,
        )
