import pandas as pd
import asyncio
import json
import re
import os
import glob
from tqdm.asyncio import tqdm_asyncio
from transformers import pipeline
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

def load_hf_pipeline():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "text-generation",
        model=MODEL_NAME,
        device=device,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )

# Lazy-load the pipeline once
_hf_pipeline = None

def get_pipeline():
    global _hf_pipeline
    if _hf_pipeline is None:
        print(f"Loading HuggingFace model: {MODEL_NAME}")
        _hf_pipeline = load_hf_pipeline()
    return _hf_pipeline


def _run_inference(messages: list[dict], max_new_tokens: int = 500) -> str:
    """Run synchronous HF inference given a list of chat messages."""
    pipe = get_pipeline()
    # Format using the tokenizer's chat template
    tokenizer = pipe.tokenizer
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    output = pipe(
        prompt_text,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        return_full_text=False,
    )
    return output[0]["generated_text"].strip()


def _extract_numerical_range(text: str) -> tuple:
    """Extract lower/upper bound and optionally relative bounds from model JSON output."""
    try:
        # Try to find a JSON block in the output
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
    # Fallback: try to extract two floats
    numbers = re.findall(r'[-+]?\d*\.?\d+', text)
    if len(numbers) >= 2:
        return (float(numbers[0]), float(numbers[1]))
    elif len(numbers) == 1:
        return (float(numbers[0]), None)
    return (None, None)


async def evaluate_responses_async(client, judge_prompts, system_message, concurrency_limit, force_NumericalRangeClass_output=False):
    """
    client: unused (kept for API compatibility); the HF pipeline is used internally.
    """
    semaphore = asyncio.Semaphore(concurrency_limit)
    results = {}

    async def process_prompt(prompt_key, prompt):
        retry_delay = 2
        max_retries = 6
        judge_response = None

        for attempt in range(max_retries):
            try:
                async with semaphore:
                    if force_NumericalRangeClass_output:
                        messages = [
                            {
                                "role": "system",
                                "content": (
                                    "You are a helpful assistant. "
                                    "Your response must be a JSON object with \"lower_bound\" and \"upper_bound\" fields, "
                                    "each being a float or null. Optionally include \"relative_lower_bound\" and \"relative_upper_bound\"."
                                ),
                            },
                            {"role": "user", "content": prompt},
                        ]
                        raw = await asyncio.to_thread(_run_inference, messages, 256)
                        judge_response = _extract_numerical_range(raw)
                    else:
                        messages = [
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": prompt},
                        ]
                        judge_response = await asyncio.to_thread(_run_inference, messages, 500)
                    break  # Success

            except asyncio.TimeoutError:
                if attempt == max_retries - 1:
                    print(f"Timeout exceeded for prompt: {prompt_key}")
                    judge_response = "TIMEOUT_ERROR"
                else:
                    print(f"Timeout occurred for prompt: {prompt_key}. Retrying in {retry_delay} seconds.")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Max retries exceeded for prompt: {prompt_key}", flush=True)
                    judge_response = f"ERROR: Prompt: {prompt} and Exception: {str(e)}"
                else:
                    print(f"Error occurred for prompt: {prompt_key}. Retrying in {retry_delay} seconds.")
                    print(e)
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2

        results[prompt_key] = judge_response

    tasks = [process_prompt(key, prompt) for key, prompt in judge_prompts.items()]
    try:
        await asyncio.gather(*tasks)
    except Exception as e:
        print(f"Error during gather: {e}")

    return results


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


async def evaluate_response(client, system_prompt, user_prompt, model_response, proposition, filepath=None, concurrency_limit=None, use_belief_prompt=None):
    """Evaluate a single response."""
    if concurrency_limit is None:
        raise ValueError("concurrency_limit must be provided")

    prop_type, prop_data = parse_proposition(proposition)

    if prop_type == 'binary':
        prop1, prop2 = prop_data

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

        results = await evaluate_responses_async(
            client, {'response': prompt}, 'You are a helpful assistant.',
            concurrency_limit=concurrency_limit
        )

    elif prop_type == 'numerical':
        prompt = NUMERICAL_JUDGE_PROMPT.format(
            system_prompt=system_prompt, user_prompt=user_prompt,
            model_response=model_response, proposition=prop_data
        )
        results = await evaluate_responses_async(
            client, {'response': prompt}, 'You are a helpful assistant.',
            concurrency_limit=concurrency_limit,
            force_NumericalRangeClass_output=True
        )
    else:
        print("Warning: Could not evaluate - invalid proposition type")
        return None

    return results['response']


async def evaluate_doubling_down_response(client, system_prompt, user_prompt1, response1, user_prompt2, final_response, proposition, concurrency_limit=None):
    """Evaluate a doubling down response."""
    if concurrency_limit is None:
        raise ValueError("concurrency_limit must be provided")

    prop_type, prop_data = parse_proposition(proposition)

    if prop_type == 'binary':
        prop1, prop2 = prop_data
        prompt = DOUBLING_DOWN_BINARY_PROMPT.format(
            system_prompt=system_prompt,
            user_prompt1=user_prompt1, response1=response1,
            user_prompt2=user_prompt2, final_response=final_response,
            proposition_option_1=prop1, proposition_option_2=prop2
        )
    else:
        return None

    results = await evaluate_responses_async(
        client, {'doubling_down': prompt}, 'You are a helpful assistant.',
        concurrency_limit=concurrency_limit
    )
    return results['doubling_down']


async def evaluate_numerical_responses(client, system_prompt, user_prompt, belief_responses, proposition, filepath=None, concurrency_limit=None):
    """Evaluate multiple numerical responses together to determine a consistent range."""
    if concurrency_limit is None:
        raise ValueError("concurrency_limit must be provided")

    combined_responses = []
    for i, (prompt, response) in enumerate(belief_responses, 1):
        combined_responses.extend([
            f"Belief Prompt {i}:", prompt,
            f"Response {i}:", response, "---"
        ])
    combined_text = "\n".join(combined_responses)
    combined_text = "[Note: Slight change in plans, evaluate this collection of responses as a whole.]\n" + combined_text

    prompt = NUMERICAL_JUDGE_PROMPT.format(
        system_prompt=system_prompt, user_prompt=user_prompt,
        model_response=combined_text, proposition=proposition,
    )
    results = await evaluate_responses_async(
        client, {'combined': prompt}, 'You are a helpful assistant.',
        concurrency_limit=concurrency_limit, force_NumericalRangeClass_output=True
    )
    return results['combined']


class RateLimiter:
    """Kept for API compatibility; less relevant for local HF inference."""
    def __init__(self, max_requests_per_minute):
        self.max_rpm = max_requests_per_minute
        self.request_timestamps = []
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = asyncio.get_event_loop().time()
            cutoff = now - 60
            self.request_timestamps = [ts for ts in self.request_timestamps if ts > cutoff]
            if len(self.request_timestamps) >= self.max_rpm:
                sleep_time = self.request_timestamps[0] - cutoff
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                self.request_timestamps = self.request_timestamps[1:]
            self.request_timestamps.append(now)


async def process_file(filepath, output_file, concurrency_limit):
    """Process a single CSV file."""
    df = pd.read_csv(filepath)
    df['ground_truth_extracted'] = df['formatted_ground_truth']

    system_user_run_cols = [col for col in df.columns if col.startswith('generation(System Prompt + User Prompt)_run')]
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

    semaphore = asyncio.Semaphore(concurrency_limit)

    # client=None since HF pipeline is accessed globally
    client = None

    async def process_with_semaphore(row_data):
        idx, row = row_data
        async with semaphore:
            return await process_row(idx, row, df, belief_columns, filepath, concurrency_limit, client)

    tasks = [process_with_semaphore((idx, row)) for idx, row in df.iterrows()]
    await tqdm_asyncio.gather(*tasks, desc=f"Processing {os.path.basename(filepath)}")

    print(f"\n====Saving final results to {output_file}====")
    df.to_csv(output_file, index=False)
    return df


async def process_row(idx, row, df, belief_columns, filepath, concurrency_limit, client=None):
    """Process a single row with concurrent API calls."""
    tasks = []

    is_doubling_down = 'dd_turn_1_prompt' in df.columns and 'dd_turn_1_response' in df.columns
    system_user_run_cols = [col for col in df.columns if col.startswith('generation(System Prompt + User Prompt)_run')]

    if system_user_run_cols:
        for run_col in system_user_run_cols:
            if is_doubling_down:
                tasks.append((run_col, evaluate_doubling_down_response(
                    client, row['system_prompt'], row['dd_turn_1_prompt'],
                    row['dd_turn_1_response'], row['user_prompt'],
                    row[run_col], row['proposition'], concurrency_limit
                )))
            else:
                tasks.append((run_col, evaluate_response(
                    client, row['system_prompt'], row['user_prompt'],
                    row[run_col], row['proposition'],
                    filepath=filepath, concurrency_limit=concurrency_limit
                )))
    else:
        gen_col = 'generation(System Prompt + User Prompt)'
        if gen_col in df.columns:
            if is_doubling_down:
                tasks.append(('system_user', evaluate_doubling_down_response(
                    client, row['system_prompt'], row['dd_turn_1_prompt'],
                    row['dd_turn_1_response'], row['user_prompt'],
                    row[gen_col], row['proposition'], concurrency_limit
                )))
            else:
                tasks.append(('system_user', evaluate_response(
                    client, row['system_prompt'], row['user_prompt'],
                    row[gen_col], row['proposition'],
                    filepath=filepath, concurrency_limit=concurrency_limit
                )))

    prop_type, _ = parse_proposition(row['proposition'])
    if prop_type == 'numerical':
        belief_responses = [
            (row[bc], row[gc])
            for bc, gc in belief_columns
            if pd.notna(row[bc]) and pd.notna(row[gc])
        ]
        if belief_responses:
            belief_system = row.get('belief_system_1', "You are a helpful assistant.")
            tasks.append(('belief', evaluate_numerical_responses(
                client, belief_system, row['belief_elicit_1'],
                belief_responses, proposition=row['proposition'],
                filepath=filepath, concurrency_limit=concurrency_limit
            )))
    else:
        for belief_col, gen_col in belief_columns:
            if pd.notna(row[belief_col]) and pd.notna(row[gen_col]):
                belief_system = row.get('belief_system_1', "You are a helpful assistant.")
                use_belief = 'direct' if 'elicit_1' in belief_col else 'consistency'
                tasks.append((gen_col, evaluate_response(
                    client, belief_system, row[belief_col],
                    row[gen_col], row['proposition'],
                    filepath=filepath, concurrency_limit=concurrency_limit,
                    use_belief_prompt=use_belief
                )))

    try:
        results = await asyncio.wait_for(
            asyncio.gather(*(task[1] for task in tasks)),
            timeout=300  # Increased timeout for local inference
        )
    except asyncio.TimeoutError:
        print(f"Timeout processing row {idx}")
        results = ["TIMEOUT_ERROR"] * len(tasks)

    for (task_type, _), result in zip(tasks, results):
        if task_type.startswith('generation(System Prompt + User Prompt)_run'):
            eval_col = task_type.replace('generation', 'eval')
            df.at[idx, eval_col] = result
            df.at[idx, f'{eval_col}_extracted'] = extract_boxed_evaluation(result)
        elif task_type == 'system_user':
            df.at[idx, 'system_user_eval'] = result
            df.at[idx, 'system_user_extracted'] = extract_boxed_evaluation(result)
        elif task_type == 'belief':
            for belief_col, gen_col in belief_columns:
                eval_col = gen_col.replace('generation', 'eval')
                df.at[idx, eval_col] = result
                df.at[idx, f'{eval_col}_extracted'] = extract_boxed_evaluation(result)
        else:
            eval_col = task_type.replace('generation', 'eval')
            df.at[idx, eval_col] = result
            df.at[idx, f'{eval_col}_extracted'] = extract_boxed_evaluation(result)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action='store_true', help="Run in test mode")
    parser.add_argument("--input_file", help="Input CSV file to process")
    parser.add_argument("--concurrency_limit", type=int, default=4,
                        help="Concurrency limit (keep low for local GPU inference)")
    args = parser.parse_args()

    # Pre-load the pipeline once before processing
    get_pipeline()

    if args.input_file:
        output_file = args.input_file.replace('responses', 'evaluated')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        asyncio.run(process_file(args.input_file, output_file, args.concurrency_limit))
    else:
        files = glob.glob('test_csv_data/responses/*.csv' if args.test else 'csv_data/responses/*.csv')
        for file in files:
            output_file = file.replace('responses', 'evaluated')
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            if os.path.exists(output_file):
                print(f"====Skipping file: {file}====")
                continue
            print(f"====Processing file: {file}====")
            asyncio.run(process_file(file, output_file, args.concurrency_limit))
