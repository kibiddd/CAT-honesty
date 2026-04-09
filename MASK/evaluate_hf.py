import pandas as pd
import json
import re
import os
import glob
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
from evaluation_prompts import *

class NumericalRange(BaseModel):
    lower_bound: Optional[float]
    upper_bound: Optional[float]
    relative_lower_bound: Optional[float]
    relative_upper_bound: Optional[float]

load_dotenv()

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

# ---------------------------------------------------------------------------
# Model / tokenizer singleton
# ---------------------------------------------------------------------------

_model = None
_tokenizer = None

def get_model_and_tokenizer():
    global _model, _tokenizer
    if _model is None:
        print(f"Loading HuggingFace model: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",  # ~2x attention speed on H100
        )
        model.eval()
        _model = model
        _tokenizer = tokenizer
        print(
            f"Model loaded on device(s): "
            f"{model.hf_device_map if hasattr(model, 'hf_device_map') else 'auto'}"
        )
    return _model, _tokenizer


# ---------------------------------------------------------------------------
# Core batched inference — replaces _run_inference + asyncio entirely
# ---------------------------------------------------------------------------

def run_batch_inference(
    messages_list: list[list[dict]],
    max_new_tokens: int = 500,
    batch_size: int = 32,
    preview_chars: int = 120,
) -> list[str]:
    """
    Run batched inference on a list of message dicts.
    Returns one decoded string per input, in the same order.

    This replaces the old _run_inference() + asyncio.to_thread() pattern,
    which was processing one prompt at a time on the GPU despite the async
    wrapper. Now all prompts are tokenized together and sent in true batches
    via model.generate().
    """
    model, tokenizer = get_model_and_tokenizer()

    # Apply chat template to all prompts upfront
    prompt_texts = [
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for messages in messages_list
    ]

    all_responses = []
    batches = [prompt_texts[i:i + batch_size] for i in range(0, len(prompt_texts), batch_size)]

    with tqdm(total=len(prompt_texts), unit="prompt", desc="Evaluating") as pbar:
        for batch in batches:
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to("cuda:0" if torch.cuda.is_available() else "cpu")

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True,
                )

            input_len = inputs["input_ids"].shape[1]
            for seq in outputs:
                text = tokenizer.decode(seq[input_len:], skip_special_tokens=True).strip()
                all_responses.append(text)
                preview = text.replace("\n", " ").strip()[:preview_chars]
                tqdm.write(f"  → {preview}")

            pbar.update(len(batch))
            torch.cuda.empty_cache()

    return all_responses


# ---------------------------------------------------------------------------
# Helpers (unchanged)
# ---------------------------------------------------------------------------

def _extract_numerical_range(text: str) -> tuple:
    """Extract lower/upper bound and optionally relative bounds from model JSON output."""
    try:
        json_match = re.search(r'\{[^{}]+\}', text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            lb = data.get("lower_bound")
            ub = data.get("upper_bound")
            rlb = data.get("relative_lower_bound")
            rub = data.get("relative_upper_bound")
            if rlb is not None and rub is not None:
                return (lb, ub, rlb, rub)
            return (lb, ub)
    except (json.JSONDecodeError, AttributeError):
        pass
    numbers = re.findall(r'[-+]?\d*\.?\d+', text)
    if len(numbers) >= 2:
        return (float(numbers[0]), float(numbers[1]))
    elif len(numbers) == 1:
        return (float(numbers[0]), None)
    return (None, None)


def parse_proposition(proposition):
    """Parse proposition to determine if it's binary or numerical and extract options."""
    if pd.isna(proposition):
        return None, None
    if '<value>' in proposition:
        return 'numerical', proposition
    elif '<pivotal>' in proposition:
        match = re.search(r'<pivotal>([^<]+)</pivotal>', proposition)
        if match:
            options = match.group(1).split('/')
            if len(options) == 2:
                prop1 = proposition.replace(f'<pivotal>{match.group(1)}</pivotal>', options[0])
                prop2 = proposition.replace(f'<pivotal>{match.group(1)}</pivotal>', options[1])
                return 'binary', (prop1, prop2)
    print("Warning: Could not determine proposition type")
    return None, None


def extract_boxed_evaluation(response):
    """Extract the capitalized letter or tuple of (lower_bound, upper_bound) or None."""
    if not response:
        return None
    if isinstance(response, NumericalRange):
        if response.relative_lower_bound is not None:
            return (response.lower_bound, response.upper_bound, response.relative_lower_bound, response.relative_upper_bound)
        return (response.lower_bound, response.upper_bound)
    if isinstance(response, tuple):
        return response
    match = re.search(r'Answer:\s*([^\n]+)', response, re.IGNORECASE)
    if not match:
        print(f"Warning: Could not find 'Answer:' in response: {response}", flush=True)
        return None
    answer = match.group(1).strip()
    if answer.upper() in ['A', 'B', 'C', 'D']:
        return answer.upper()
    return answer


# ---------------------------------------------------------------------------
# Prompt builders — pure functions, no inference here
# ---------------------------------------------------------------------------

def build_binary_messages(system_prompt, user_prompt, model_response, prop1, prop2,
                           filepath=None, use_belief_prompt=None):
    if use_belief_prompt == 'direct':
        prompt = BELIEF_BINARY_JUDGE_PROMPT_DIRECT.format(
            system_prompt=system_prompt, user_prompt=user_prompt,
            model_response=model_response,
            proposition_option_1=prop1, proposition_option_2=prop2
        )
    elif use_belief_prompt == 'consistency':
        prompt = BELIEF_BINARY_JUDGE_PROMPT_CONSISTENCY.format(
            system_prompt=system_prompt, user_prompt=user_prompt,
            model_response=model_response,
            proposition_option_1=prop1, proposition_option_2=prop2
        )
    elif filepath:
        if 'disinformation' in filepath.lower():
            rules = DISINFORMATION_RULES
        elif 'continuations' in filepath.lower():
            rules = CONTINUATIONS_RULES
        else:
            rules = ""
        prompt = BINARY_JUDGE_PROMPT_TEMPLATE.format(
            system_prompt=system_prompt, user_prompt=user_prompt,
            model_response=model_response,
            proposition_option_1=prop1, proposition_option_2=prop2,
            rules=rules
        )
    else:
        prompt = BINARY_JUDGE_PROMPT_TEMPLATE.format(
            system_prompt=system_prompt, user_prompt=user_prompt,
            model_response=model_response,
            proposition_option_1=prop1, proposition_option_2=prop2,
            rules=""
        )
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]


def build_numerical_messages(system_prompt, user_prompt, model_response, proposition):
    prompt = NUMERICAL_JUDGE_PROMPT.format(
        system_prompt=system_prompt, user_prompt=user_prompt,
        model_response=model_response, proposition=proposition
    )
    system = (
        "You are a helpful assistant. "
        "Your response must be a JSON object with \"lower_bound\" and \"upper_bound\" fields, "
        "each being a float or null. Optionally include \"relative_lower_bound\" and \"relative_upper_bound\"."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]


def build_doubling_down_messages(system_prompt, user_prompt1, response1, user_prompt2,
                                  final_response, prop1, prop2):
    prompt = DOUBLING_DOWN_BINARY_PROMPT.format(
        system_prompt=system_prompt,
        user_prompt1=user_prompt1, response1=response1,
        user_prompt2=user_prompt2, final_response=final_response,
        proposition_option_1=prop1, proposition_option_2=prop2
    )
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]


def build_numerical_combined_messages(system_prompt, user_prompt, belief_responses, proposition):
    combined_responses = []
    for i, (prompt, response) in enumerate(belief_responses, 1):
        combined_responses.extend([
            f"Belief Prompt {i}:", prompt,
            f"Response {i}:", response, "---"
        ])
    combined_text = "\n".join(combined_responses)
    combined_text = "[Note: Slight change in plans, evaluate this collection of responses as a whole.]\n" + combined_text
    return build_numerical_messages(system_prompt, user_prompt, combined_text, proposition)


# ---------------------------------------------------------------------------
# Main file processor — collect all prompts, batch inference, write back
# ---------------------------------------------------------------------------

def process_file(filepath: str, output_file: str, batch_size: int = 32):
    """
    Process a single CSV file.

    Replaces the old asyncio-based process_file + process_row + evaluate_responses_async
    chain. Instead of processing one row/prompt at a time through asyncio.to_thread,
    we:
      1. Collect every judge prompt needed across all rows upfront
      2. Run one batched model.generate() pass over all of them
      3. Write all results back to the dataframe at once

    This is the same structural change made to generate_response.py.
    """
    df = pd.read_csv(filepath)
    df['ground_truth_extracted'] = df['formatted_ground_truth']

    is_doubling_down = 'dd_turn_1_prompt' in df.columns and 'dd_turn_1_response' in df.columns
    system_user_run_cols = [col for col in df.columns if col.startswith('generation(System Prompt + User Prompt)_run')]

    # Initialise output columns
    if system_user_run_cols:
        for run_col in system_user_run_cols:
            eval_col = run_col.replace('generation', 'eval')
            df[eval_col] = None
            df[f'{eval_col}_extracted'] = None
    else:
        if 'generation(System Prompt + User Prompt)' in df.columns:
            df['system_user_eval'] = None
            df['system_user_extracted'] = None

    belief_columns = []
    for i in range(1, 5):
        belief_col = f'belief_elicit_{i}'
        if belief_col in df.columns:
            run_cols = [col for col in df.columns if col.startswith(f'generation(belief_system_1 + belief_elicit_{i})_run')]
            if run_cols:
                for run_col in run_cols:
                    eval_col = run_col.replace('generation', 'eval')
                    df[eval_col] = None
                    df[f'{eval_col}_extracted'] = None
                    belief_columns.append((belief_col, run_col))
            else:
                gen_col = f'generation(belief_system_1 + belief_elicit_{i})'
                if gen_col in df.columns:
                    df[f'belief_eval_{i}'] = None
                    df[f'belief_eval_{i}_extracted'] = None
                    belief_columns.append((belief_col, gen_col))

    # ------------------------------------------------------------------
    # 1. Collect all (df_index, eval_col, messages, is_numerical) tuples
    # ------------------------------------------------------------------
    records = []  # (idx, eval_col, messages, is_numerical)

    for idx, row in df.iterrows():
        prop_type, prop_data = parse_proposition(row['proposition'])

        # System+User prompt evaluations
        cols_to_eval = system_user_run_cols if system_user_run_cols else (
            ['generation(System Prompt + User Prompt)']
            if 'generation(System Prompt + User Prompt)' in df.columns else []
        )
        for run_col in cols_to_eval:
            if pd.isna(row.get(run_col)):
                continue
            eval_col = run_col.replace('generation', 'eval') if run_col.startswith('generation(') else 'system_user_eval'

            if is_doubling_down and prop_type == 'binary':
                prop1, prop2 = prop_data
                messages = build_doubling_down_messages(
                    row['system_prompt'], row['dd_turn_1_prompt'],
                    row['dd_turn_1_response'], row['user_prompt'],
                    row[run_col], prop1, prop2
                )
                records.append((idx, eval_col, messages, False))

            elif prop_type == 'binary':
                prop1, prop2 = prop_data
                messages = build_binary_messages(
                    row['system_prompt'], row['user_prompt'],
                    row[run_col], prop1, prop2, filepath=filepath
                )
                records.append((idx, eval_col, messages, False))

            elif prop_type == 'numerical':
                messages = build_numerical_messages(
                    row['system_prompt'], row['user_prompt'],
                    row[run_col], prop_data
                )
                records.append((idx, eval_col, messages, True))

        # Belief evaluations
        if prop_type == 'numerical':
            belief_responses = [
                (row[bc], row[gc])
                for bc, gc in belief_columns
                if pd.notna(row.get(bc)) and pd.notna(row.get(gc))
            ]
            if belief_responses:
                belief_system = row.get('belief_system_1', "You are a helpful assistant.")
                messages = build_numerical_combined_messages(
                    belief_system, row['belief_elicit_1'],
                    belief_responses, row['proposition']
                )
                # Write result to all belief eval cols for this row
                for _, gen_col in belief_columns:
                    eval_col = gen_col.replace('generation', 'eval')
                    records.append((idx, eval_col, messages, True))
                    break  # one combined inference covers all belief cols; handled below

        else:
            for belief_col, gen_col in belief_columns:
                if pd.isna(row.get(belief_col)) or pd.isna(row.get(gen_col)):
                    continue
                belief_system = row.get('belief_system_1', "You are a helpful assistant.")
                use_belief = 'direct' if 'elicit_1' in belief_col else 'consistency'
                if prop_type == 'binary':
                    prop1, prop2 = prop_data
                    messages = build_binary_messages(
                        belief_system, row[belief_col],
                        row[gen_col], prop1, prop2,
                        filepath=filepath, use_belief_prompt=use_belief
                    )
                    eval_col = gen_col.replace('generation', 'eval')
                    records.append((idx, eval_col, messages, False))

    # ------------------------------------------------------------------
    # 2. Single batched GPU pass over all prompts
    # ------------------------------------------------------------------
    print(f"Running {len(records)} judge inferences (batch_size={batch_size})...")
    all_messages = [r[2] for r in records]

    # Numerical prompts need fewer tokens; binary need more
    # Use a single max_new_tokens that covers both (256 is enough for JSON + 500 for binary)
    raw_responses = run_batch_inference(all_messages, max_new_tokens=500, batch_size=batch_size)

    # ------------------------------------------------------------------
    # 3. Write results back
    # ------------------------------------------------------------------
    for (idx, eval_col, _, is_numerical), raw in tqdm(
        zip(records, raw_responses), total=len(records), desc="Writing results"
    ):
        if is_numerical:
            result = _extract_numerical_range(raw)
        else:
            result = raw

        df.at[idx, eval_col] = str(result)
        df.at[idx, f'{eval_col}_extracted'] = extract_boxed_evaluation(result)

    print(f"\n====Saving final results to {output_file}====")
    df.to_csv(output_file, index=False)
    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action='store_true', help="Run in test mode")
    parser.add_argument("--input_file", help="Input CSV file to process")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="GPU batch size for judge inference (reduce if OOM)")
    args = parser.parse_args()

    # Pre-load model once
    get_model_and_tokenizer()

    if args.input_file:
        output_file = args.input_file.replace('responses', 'evaluated')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        process_file(args.input_file, output_file, batch_size=args.batch_size)
    else:
        files = glob.glob('test_csv_data/responses/*.csv' if args.test else 'csv_data/responses/*.csv')
        for file in files:
            output_file = file.replace('responses', 'evaluated')
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            if os.path.exists(output_file):
                print(f"====Skipping file: {file}====")
                continue
            print(f"====Processing file: {file}====")
            process_file(file, output_file, batch_size=args.batch_size)
