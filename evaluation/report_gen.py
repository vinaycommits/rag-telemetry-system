# evaluation/report_gen.py

import json
import statistics
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def load_results(path: str = "evaluation/results.jsonl") -> list[dict]:
    results = []
    with open(path) as f:
        for line in f:
            results.append(json.loads(line))
    return results


def plot_overall_scores(results: list[dict], out_dir: Path):
    """Bar chart: score per question, coloured by pass/fail."""

    ids     = [r["question_id"] for r in results]
    scores  = [r["score"] for r in results]
    colors  = ["#2ecc71" if s >= 0.6 else "#e74c3c" for s in scores]

    fig, ax = plt.subplots(figsize=(14, 5))
    bars    = ax.bar(ids, scores, color=colors, edgecolor="white", linewidth=0.5)

    ax.axhline(0.6,  color="gray",   linestyle="--", linewidth=1, label="Pass threshold (0.6)")
    ax.axhline(statistics.mean(scores), color="steelblue",
               linestyle="-", linewidth=1.5,
               label=f"Mean score ({statistics.mean(scores):.2f})")

    ax.set_title("Score per Question",  fontsize=14, fontweight="bold")
    ax.set_xlabel("Question ID")
    ax.set_ylabel("Score (0-1)")
    ax.set_ylim(0, 1.1)
    ax.tick_params(axis="x", rotation=45)
    ax.legend()

    green_patch = mpatches.Patch(color="#2ecc71", label="Pass (≥0.6)")
    red_patch   = mpatches.Patch(color="#e74c3c", label="Fail (<0.6)")
    ax.legend(handles=[green_patch, red_patch,
                        plt.Line2D([0], [0], color="steelblue", linewidth=1.5,
                                   label=f"Mean ({statistics.mean(scores):.2f})")])

    plt.tight_layout()
    plt.savefig(out_dir / "01_scores_per_question.png", dpi=150)
    plt.close()
    print("Saved: 01_scores_per_question.png")


def plot_tag_breakdown(results: list[dict], out_dir: Path):
    """Horizontal bar chart: average score per failure mode tag."""

    tag_scores: dict[str, list[float]] = defaultdict(list)
    for r in results:
        for tag in r["tags"]:
            tag_scores[tag].append(r["score"])

    tags   = list(tag_scores.keys())
    means  = [statistics.mean(tag_scores[t]) for t in tags]
    colors = ["#2ecc71" if m >= 0.6 else "#e74c3c" for m in means]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(tags, means, color=colors, edgecolor="white")
    ax.axvline(0.6, color="gray", linestyle="--", linewidth=1, label="Pass threshold")
    ax.set_title("Average Score by Failure Mode", fontsize=14, fontweight="bold")
    ax.set_xlabel("Average Score (0-1)")
    ax.set_xlim(0, 1.1)

    for i, (tag, mean) in enumerate(zip(tags, means)):
        ax.text(mean + 0.02, i, f"{mean:.2f}", va="center", fontsize=10)

    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "02_tag_breakdown.png", dpi=150)
    plt.close()
    print("Saved: 02_tag_breakdown.png")


def plot_answer_type_comparison(results: list[dict], out_dir: Path):
    """Grouped bar: exact match vs LLM judge scores."""

    exact      = [r["score"] for r in results if r["answer_type"] == "exact"]
    open_ended = [r["score"] for r in results if r["answer_type"] == "open_ended"]

    labels = ["Exact Match", "LLM Judge (Open-ended)"]
    means  = [
        statistics.mean(exact)      if exact      else 0,
        statistics.mean(open_ended) if open_ended else 0,
    ]
    colors = ["#3498db", "#9b59b6"]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars    = ax.bar(labels, means, color=colors, width=0.4, edgecolor="white")
    ax.axhline(0.6, color="gray", linestyle="--", linewidth=1, label="Pass threshold")
    ax.set_title("Exact Match vs LLM Judge", fontsize=14, fontweight="bold")
    ax.set_ylabel("Average Score (0-1)")
    ax.set_ylim(0, 1.1)

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                mean + 0.03, f"{mean:.2f}", ha="center", fontsize=12)

    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "03_answer_type_comparison.png", dpi=150)
    plt.close()
    print("Saved: 03_answer_type_comparison.png")


def plot_latency(results: list[dict], out_dir: Path):
    """Line chart: retrieval latency per question."""

    ids          = [r["question_id"] for r in results]
    retrieval_ms = [r["retrieval_ms"] for r in results]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(ids, retrieval_ms, marker="o", color="steelblue", linewidth=1.5)
    ax.axhline(statistics.mean(retrieval_ms), color="orange", linestyle="--",
               linewidth=1, label=f"Mean ({statistics.mean(retrieval_ms):.1f}ms)")
    ax.set_title("Retrieval Latency per Question", fontsize=14, fontweight="bold")
    ax.set_xlabel("Question ID")
    ax.set_ylabel("Latency (ms)")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "04_latency.png", dpi=150)
    plt.close()
    print("Saved: 04_latency.png")


def generate_report(results_path: str = "evaluation/results.jsonl"):
    results = load_results(results_path)
    out_dir = Path("evaluation/charts")
    out_dir.mkdir(exist_ok=True)

    print(f"\nGenerating report for {len(results)} results...\n")

    plot_overall_scores(results, out_dir)
    plot_tag_breakdown(results, out_dir)
    plot_answer_type_comparison(results, out_dir)
    plot_latency(results, out_dir)

    print(f"\nAll charts saved to {out_dir}/")


if __name__ == "__main__":
    generate_report()