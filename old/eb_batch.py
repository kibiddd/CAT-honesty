"""
elicit_belief.py

Runs the target model on the MASK belief elicitation prompts (belief_elicit_1/2/3)
for each row in the specified MASK subsets. Saves raw model responses to a CSV.

Subsets used: continuations, known_facts, statistics
Skipped:      doubling_down (different schema), provided_facts (huge system prompts)

Output: data/interim/belief_responses_{subset}.csv
  Contains all original MASK columns + generated response columns:
    generation(belief_system_1 + belief_elicit_1)_run1
    generation(belief_system_1 + belief_elicit_1)_run2
    generation(belief_system_1 + belief_elicit_1)_run3
    generation(belief_system_1 + belief_elicit_2)_run1
    generation(belief_system_1 + belief_elicit_3)_run1
  (Column naming matches MASK's evaluate.py expectations exactly so it can
   be passed directly to 2_evaluate_responses.py.)

Usage:
    python elicit_belief.py --model meta-llama/Llama-3.1-8B-Instruct
    python elicit_belief.py --model meta-llama/Llama-3.1-8B-Instruct --subset known_facts
    python elicit_belief.py --model meta-llama/Llama-3.1-8B-Instruct --resume
    python elicit_belief.py --model meta-llama/Llama-3.1-8B-Instruct --batch_size 4
"""

import argparse
import logging
import os
import sys

import pandas as pd
import torch
import yaml
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── Column naming must match MASK's evaluate.py exactly ──────────────────────
# belief_elicit_1 is run 3 times (majority vote), 2 and 3 once each
BELIEF_K = {"1": 3, "2": 1, "3": 1}

SUBSETS = ["continuations", "known_facts", "statistics"]

BELIEF_ELICIT_COLS = {
    "continuations": ["belief_elicit_1", "belief_elicit_2", "belief_elicit_3"],
    "known_facts":   ["belief_elicit_1", "belief_elicit_2", "belief_elicit_3"],
    "statistics":    ["belief_elicit_1"],
}


def setup_logger(output_dir: str, model_shortname: str) -> logging.Logger:
    logger = logging.getLogger("elicit_belief")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler → stdout so it doesn't mix with tqdm's stderr
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    # File handler → dedicated log per run
    log_path = os.path.join(output_dir, f"elicit_{model_shortname}.log")
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_mask_subset(subset: str, logger: logging.Logger) -> pd.DataFrame:
    """Load a MASK subset from HuggingFace and return as DataFrame."""
    logger.info(f"Loading MASK/{subset} from HuggingFace...")
    ds = load_dataset("cais/MASK", subset, split="test")
    df = ds.to_pandas()
    if "belief_system_1" not in df.columns:
        df["belief_system_1"] = "You are a helpful assistant."
    return df


def build_chat_messages(system_prompt: str, user_prompt: str) -> list:
    """Format into the standard chat messages list."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]


def generate_batch(
    model,
    tokenizer,
    messages_list: list,
    max_new_tokens: int = 1000,
    device: str = "cuda",
) -> list[str]:
    """Run a batch of chat prompts and return a list of response strings."""
    inputs = tokenizer.apply_chat_template(
        messages_list,
        add_generation_prompt=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_len = inputs.shape[-1]
    return [
        tokenizer.decode(out[input_len:], skip_special_tokens=True).strip()
        for out in outputs
    ]


def process_subset(
    subset: str,
    model,
    tokenizer,
    output_dir: str,
    model_shortname: str,
    logger: logging.Logger,
    max_new_tokens: int = 1000,
    resume: bool = True,
    device: str = "cuda",
    batch_size: int = 4,
) -> str:
    """
    Run belief elicitation for one MASK subset.
    Returns path to the output CSV.
    """
    output_path = os.path.join(output_dir, f"belief_responses_{subset}_{model_shortname}.csv")

    df = load_mask_subset(subset, logger)
    belief_cols = BELIEF_ELICIT_COLS[subset]

    expected_cols = []
    for be_num_str, k in BELIEF_K.items():
        be_col = f"belief_elicit_{be_num_str}"
        if be_col not in belief_cols:
            continue
        for run in range(1, k + 1):
            expected_cols.append(
                f"generation(belief_system_1 + belief_elicit_{be_num_str})_run{run}"
            )

    if resume and os.path.exists(output_path):
        existing = pd.read_csv(output_path)
        done_mask = existing[expected_cols].notna().all(axis=1)
        n_done = done_mask.sum()
        logger.info(f"[{subset}] Resuming: {n_done}/{len(df)} rows already done.")
        df = df[~done_mask].copy().reset_index(drop=True)
    else:
        existing = None
        for col in expected_cols:
            df[col] = None

    if len(df) == 0:
        logger.warning(f"[{subset}] All rows complete, skipping.")
        return output_path

    logger.info(f"[{subset}] Processing {len(df)} rows with batch_size={batch_size}...")

    # ── Build flat task list: (idx, out_col, messages) ───────────────────────
    tasks = []
    for idx, row in df.iterrows():
        system_prompt = row.get("belief_system_1", "You are a helpful assistant.")
        for be_num_str, k in BELIEF_K.items():
            be_col = f"belief_elicit_{be_num_str}"
            if be_col not in belief_cols or pd.isna(row.get(be_col)):
                continue
            user_prompt = row[be_col]
            messages = build_chat_messages(system_prompt, user_prompt)
            for run in range(1, k + 1):
                out_col = f"generation(belief_system_1 + belief_elicit_{be_num_str})_run{run}"
                tasks.append((idx, out_col, messages))

    logger.info(f"[{subset}] Total inference calls: {len(tasks)} "
                f"({len(tasks) // batch_size + 1} batches)")

    # ── Process in batches ────────────────────────────────────────────────────
    checkpoint_every = max(1, 50 // batch_size)  # checkpoint roughly every ~50 rows

    for batch_idx, batch_start in enumerate(
        tqdm(range(0, len(tasks), batch_size), desc=subset)
    ):
        batch = tasks[batch_start : batch_start + batch_size]
        messages_list = [t[2] for t in batch]

        try:
            responses = generate_batch(
                model, tokenizer, messages_list,
                max_new_tokens=max_new_tokens,
                device=device,
            )
        except RuntimeError as e:
            logger.error(f"[{subset}] OOM or runtime error at batch {batch_idx}: {e}")
            logger.warning(f"[{subset}] Skipping batch {batch_idx}, filling with None.")
            responses = [None] * len(batch)

        for (idx, out_col, _), response in zip(batch, responses):
            df.at[idx, out_col] = response

        if (batch_idx + 1) % checkpoint_every == 0:
            _save(df, existing, output_path, expected_cols, logger)

    # Final save
    _save(df, existing, output_path, expected_cols, logger)
    logger.info(f"[{subset}] Done. Saved to {output_path}")
    return output_path


def _save(df_new, df_existing, path, expected_cols, logger: logging.Logger):
    """Save only completed rows, merged with previously completed rows."""
    completed_new = df_new[df_new[expected_cols].notna().all(axis=1)]

    if df_existing is not None:
        done_existing = df_existing[df_existing[expected_cols].notna().all(axis=1)]
        out = pd.concat([done_existing, completed_new], ignore_index=True)
    else:
        out = completed_new

    if len(out) > 0:
        out.to_csv(path, index=False)
        logger.debug(f"Checkpoint saved: {len(out)} rows → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--subset", default=None, choices=SUBSETS + [None])
    parser.add_argument("--max_new_tokens", type=int, default=1000)
    parser.add_argument("--output_dir", default="data/interim")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no_resume", dest="resume", action="store_false")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Inference batch size (default: 4, reduce if OOM)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # model_shortname must be defined before setup_logger
    model_shortname = args.model.split("/")[-1]
    logger = setup_logger(args.output_dir, model_shortname)

    logger.info(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    logger.info(f"Model loaded. batch_size={args.batch_size}, "
                f"max_new_tokens={args.max_new_tokens}")

    subsets_to_run = [args.subset] if args.subset else SUBSETS

    for subset in subsets_to_run:
        process_subset(
            subset=subset,
            model=model,
            tokenizer=tokenizer,
            output_dir=args.output_dir,
            model_shortname=model_shortname,
            logger=logger,
            max_new_tokens=args.max_new_tokens,
            resume=args.resume,
            device=args.device,
            batch_size=args.batch_size,
        )

    logger.info("All done.")


if __name__ == "__main__":
    main()
