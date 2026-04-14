#!/usr/bin/env python3
"""
Parse all_results.json and generate comparative plots showing individual
model configurations' performance relative to the original baseline ("orig").

Usage:
    python3 plot_results.py <results.json> [output_dir]

Two base models: Llama 8B and Llama 70B
Six tasks: disinformation, continuations, statistics, known_facts,
           doubling_down_known_facts, provided_facts_eval
Metrics plotted: accurate, honest_1, honesty_score_1
"""

import json
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

matplotlib.use("Agg")

# Constants 

TASKS = [
    "disinformation",
    "continuations",
    "statistics",
    "known_facts",
    "doubling_down_known_facts",
    "provided_facts_eval",
]

TASK_LABELS = {
    "disinformation": "Disinfo.",
    "continuations": "Contin.",
    "statistics": "Statistics",
    "known_facts": "Known Facts",
    "doubling_down_known_facts": "Doubling Down",
    "provided_facts_eval": "Provided Facts",
}

METRICS = ["accurate", "honest_1", "honesty_score_1"]
METRIC_LABELS = {
    "accurate": "Accuracy",
    "honest_1": "Honest₁",
    "honesty_score_1": "Honesty Score₁",
}


# Parsing

def classify_model(key: str) -> str | None:
    if "8B" in key or "8b" in key:
        return "8B"
    if "70B" in key or "70b" in key:
        return "70B"
    return None


def classify_task(key: str) -> str | None:
    for task in sorted(TASKS, key=len, reverse=True):
        if key.startswith(task):
            return task
    return None


def classify_config_type(config: str) -> str:
    if config == "orig":
        return "orig"
    if config.startswith("dpo"):
        return "dpo"
    if config.startswith("ul"):
        return "ul"
    return "other"


def parse_results(filepath: str) -> pd.DataFrame:
    """Parse JSON into a tidy DataFrame with one row per (task, model, config)."""
    with open(filepath) as f:
        data = json.load(f)

    rows = []
    for top_key, configs in data.items():
        if top_key.endswith("_ul"):
            continue

        task = classify_task(top_key)
        model = classify_model(top_key)
        if task is None or model is None:
            continue

        # Skip non-eval provided_facts (274 responses)
        if "provided_facts" in top_key and "eval" not in top_key:
            continue

        for config, metrics in configs.items():
            if task == "provided_facts_eval":
                if metrics.get("total_responses") != 174:
                    continue

            ct = classify_config_type(config)
            row = {
                "model": model,
                "task": task,
                "config": config,
                "config_type": ct,
            }
            for m in METRICS:
                row[m] = metrics.get(m, None)
            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def compute_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Compute delta of each metric relative to 'orig' for each (model, task)."""
    orig = df[df["config"] == "orig"].groupby(["model", "task"])[METRICS].mean()

    records = []
    for _, row in df.iterrows():
        if row["config"] == "orig":
            continue
        key = (row["model"], row["task"])
        if key not in orig.index:
            continue
        orig_vals = orig.loc[key]
        rec = row.to_dict()
        for m in METRICS:
            ov, rv = orig_vals[m], row[m]
            rec[f"{m}_delta"] = rv - ov if pd.notna(rv) and pd.notna(ov) else None
            rec[f"{m}_orig"] = ov
        records.append(rec)

    return pd.DataFrame(records)


# ── Plotting ─────────────────────────────────────────────────────────────────

DPO_COLORS = [
    "#1565C0", "#1E88E5", "#42A5F5", "#64B5F6", "#90CAF9",
    "#BBDEFB", "#0D47A1", "#2979FF", "#448AFF", "#82B1FF",
]
UL_COLORS = [
    "#E65100", "#F57C00", "#FF9800", "#FFB74D", "#FFCC80",
    "#FFE0B2", "#BF360C", "#FF6D00", "#FF9100", "#FFAB40",
]


def get_config_color(config: str, configs_sorted: list, config_type: str) -> str:
    palette = DPO_COLORS if config_type == "dpo" else UL_COLORS
    idx = configs_sorted.index(config) if config in configs_sorted else 0
    return palette[idx % len(palette)]


def plot_heatmap_per_group(delta_df: pd.DataFrame, output_dir: Path):
    """
    Heatmap: rows = individual configs, columns = tasks.
    One figure per (model, config_type) with 3 metric sub-panels.
    Best config per task column highlighted with gold border.
    """
    for model in ["8B", "70B"]:
        for ct in ["dpo", "ul"]:
            subset = delta_df[(delta_df["model"] == model) & (delta_df["config_type"] == ct)]
            if subset.empty:
                continue

            configs = sorted(subset["config"].unique())
            tasks_present = [t for t in TASKS if t in subset["task"].values]

            fig, axes = plt.subplots(1, 3, figsize=(20, max(3, len(configs) * 0.5 + 1.5)))
            fig.suptitle(
                f"Llama {model} — {ct.upper()} Configs — Δ vs. Original",
                fontsize=14, fontweight="bold", y=1.02,
            )

            for mi, metric in enumerate(METRICS):
                ax = axes[mi]
                col = f"{metric}_delta"

                matrix = np.full((len(configs), len(tasks_present)), np.nan)
                for ci, config in enumerate(configs):
                    for ti, task in enumerate(tasks_present):
                        row = subset[(subset["config"] == config) & (subset["task"] == task)]
                        if len(row) > 0 and pd.notna(row[col].values[0]):
                            matrix[ci, ti] = row[col].values[0]

                valid = matrix[~np.isnan(matrix)]
                if len(valid) == 0:
                    ax.set_visible(False)
                    continue
                vmax = max(abs(valid.max()), abs(valid.min()), 1)

                im = ax.imshow(matrix, cmap="RdBu_r", aspect="auto",
                               vmin=-vmax, vmax=vmax)

                ax.set_xticks(range(len(tasks_present)))
                ax.set_xticklabels([TASK_LABELS.get(t, t) for t in tasks_present],
                                   fontsize=8, rotation=30, ha="right")
                ax.set_yticks(range(len(configs)))
                ax.set_yticklabels(configs, fontsize=8)
                ax.set_title(METRIC_LABELS[metric], fontsize=11, fontweight="bold")

                # Best config per task (handle all-NaN columns)
                col_best = np.full(len(tasks_present), -1, dtype=int)
                for ti in range(len(tasks_present)):
                    col_vals = matrix[:, ti]
                    if not np.all(np.isnan(col_vals)):
                        col_best[ti] = np.nanargmax(col_vals)

                for ci in range(len(configs)):
                    for ti in range(len(tasks_present)):
                        val = matrix[ci, ti]
                        if np.isnan(val):
                            continue
                        txt_color = "white" if abs(val) > vmax * 0.6 else "black"
                        weight = "bold" if ci == col_best[ti] else "normal"
                        ax.text(ti, ci, f"{val:+.1f}", ha="center", va="center",
                                fontsize=7, fontweight=weight, color=txt_color)
                        if ci == col_best[ti]:
                            ax.add_patch(plt.Rectangle(
                                (ti - 0.5, ci - 0.5), 1, 1,
                                fill=False, edgecolor="gold", linewidth=2.5,
                            ))

                plt.colorbar(im, ax=ax, shrink=0.8)

            fig.tight_layout()
            fname = output_dir / f"heatmap_{model}_{ct}.png"
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {fname}")


def plot_per_config_bars(delta_df: pd.DataFrame, output_dir: Path):
    """
    Grouped bar chart: individual configs as separate bars, grouped by task.
    One figure per (model, config_type) with 3 metric sub-panels.
    """
    for model in ["8B", "70B"]:
        for ct in ["dpo", "ul"]:
            subset = delta_df[(delta_df["model"] == model) & (delta_df["config_type"] == ct)]
            if subset.empty:
                continue

            configs = sorted(subset["config"].unique())
            tasks_present = [t for t in TASKS if t in subset["task"].values]
            n_configs = len(configs)
            n_tasks = len(tasks_present)

            fig, axes = plt.subplots(1, 3, figsize=(18, 5 + n_configs * 0.15))
            fig.suptitle(
                f"Llama {model} — {ct.upper()} Configs vs. Original Baseline",
                fontsize=14, fontweight="bold", y=1.02,
            )

            for mi, metric in enumerate(METRICS):
                ax = axes[mi]
                col = f"{metric}_delta"

                x = np.arange(n_tasks)
                total_width = 0.8
                bar_width = total_width / n_configs

                for ci, config in enumerate(configs):
                    vals = []
                    for task in tasks_present:
                        row = subset[(subset["config"] == config) & (subset["task"] == task)]
                        vals.append(row[col].values[0] if len(row) > 0 and pd.notna(row[col].values[0]) else 0)

                    offset = (ci - n_configs / 2 + 0.5) * bar_width
                    color = get_config_color(config, configs, ct)
                    ax.bar(x + offset, vals, bar_width * 0.9,
                           label=config, color=color,
                           edgecolor="white", linewidth=0.5)

                ax.axhline(0, color="black", linewidth=0.8)
                ax.set_xticks(x)
                ax.set_xticklabels([TASK_LABELS.get(t, t) for t in tasks_present],
                                   fontsize=9, rotation=30, ha="right")
                ax.set_ylabel(f"Δ {METRIC_LABELS[metric]}", fontsize=10)
                ax.set_title(METRIC_LABELS[metric], fontsize=11, fontweight="bold")
                ax.grid(axis="y", alpha=0.3, linewidth=0.5)
                if mi == len(METRICS) - 1:
                    ax.legend(fontsize=7, ncol=2, loc="upper right")

            fig.tight_layout()
            fname = output_dir / f"bars_{model}_{ct}.png"
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {fname}")


def plot_accuracy_vs_honesty(delta_df: pd.DataFrame, output_dir: Path):
    """
    Scatter: Δ Accuracy vs Δ Honesty Score₁ for each config.
    Each dot = one (config, task) pair. Diamond = config mean.
    Helps find configs that improve honesty without sacrificing accuracy.
    """
    groups = list(delta_df.groupby(["model", "config_type"]))
    n_groups = len(groups)
    if n_groups == 0:
        return

    fig, axes = plt.subplots(1, n_groups, figsize=(6 * n_groups, 5))
    if n_groups == 1:
        axes = [axes]

    for ax, ((model, ct), grp) in zip(axes, groups):
        configs = sorted(grp["config"].unique())

        for config in configs:
            cfg_data = grp[grp["config"] == config]
            acc = cfg_data["accurate_delta"].values
            hon = cfg_data["honesty_score_1_delta"].values
            mask = ~(np.isnan(acc) | np.isnan(hon))
            if mask.sum() == 0:
                continue

            color = get_config_color(config, configs, ct)
            ax.scatter(acc[mask], hon[mask], color=color, s=50, alpha=0.7,
                       edgecolors="white", linewidth=0.5)
            ax.scatter(np.nanmean(acc[mask]), np.nanmean(hon[mask]),
                       color=color, s=150, marker="D", edgecolors="black",
                       linewidth=1.5, label=config, zorder=5)

        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_xlabel("Δ Accuracy", fontsize=10)
        ax.set_ylabel("Δ Honesty Score₁", fontsize=10)
        ax.set_title(f"Llama {model} — {ct.upper()}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.2)

    fig.suptitle("Accuracy–Honesty Tradeoff (Δ vs. Original)\nDiamonds = config mean across tasks",
                 fontsize=13, fontweight="bold", y=1.04)
    fig.tight_layout()
    fname = output_dir / "tradeoff_accuracy_honesty.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    filepath = sys.argv[1] if len(sys.argv) > 1 else "csv_data/metrics/all_results.json"
    output_dir = Path(sys.argv[2] if len(sys.argv) > 2 else ".")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {filepath}...")
    df = parse_results(filepath)
    print(f"Parsed {len(df)} rows: {df['model'].nunique()} models, "
          f"{df['task'].nunique()} tasks, {df['config'].nunique()} configs")

    delta_df = compute_deltas(df)
    print(f"{len(delta_df)} delta rows (excluding orig baselines)")

    orig = df[df["config"] == "orig"].groupby(["model", "task"])[METRICS].mean().reset_index()
    print("\n── Original Baselines ──")
    print(orig.to_string(index=False))

    print("\n── Generating Plots ──")
    plot_heatmap_per_group(delta_df, output_dir)
    plot_per_config_bars(delta_df, output_dir)
    plot_accuracy_vs_honesty(delta_df, output_dir)
    print("\nDone!")


if __name__ == "__main__":
    main()
