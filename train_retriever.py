import argparse
import json
from pathlib import Path
from statistics import mean

import numpy as np
from sentence_transformers import InputExample, SentenceTransformer, losses
import torch
from torch.utils.data import DataLoader
from transformers.optimization import get_linear_schedule_with_warmup

from runtime_env import ensure_model_cache_dirs
from search_profiles import (
    DEFAULT_PROFILE_KEY,
    get_profile,
    prepare_document_text,
    prepare_query_text,
)

BASE_DIR = Path(__file__).resolve().parent
TRAIN_DIR = BASE_DIR / "data" / "training"
MODEL_ROOT = BASE_DIR / "models" / "retriever" / "minilm-domain-ft"
REPORT_DIR = BASE_DIR / "reports" / "training"

DEFAULT_CORPUS = TRAIN_DIR / "retriever_corpus.jsonl"
DEFAULT_PAIRS = TRAIN_DIR / "retriever_pairs.jsonl"
DEFAULT_TRIPLETS = TRAIN_DIR / "retriever_triplets.jsonl"
DEFAULT_OUTPUT_DIR = MODEL_ROOT


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a domain-adaptive dataset retriever from training pairs/triplets."
    )
    parser.add_argument("--base-profile", default=DEFAULT_PROFILE_KEY)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--triplets", type=Path, default=DEFAULT_TRIPLETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pair-epochs", type=int, default=2)
    parser.add_argument("--triplet-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--triplet-margin", type=float, default=0.25)
    parser.add_argument("--eval-top-k", type=int, default=5)
    parser.add_argument("--skip-pair-phase", action="store_true")
    parser.add_argument("--skip-triplet-phase", action="store_true")
    return parser.parse_args()


def iter_jsonl(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc


def load_jsonl(path: Path):
    return list(iter_jsonl(path))


def build_corpus_lookup(rows: list[dict], profile_key: str):
    lookup = {}
    for row in rows:
        ref = row.get("ref")
        if not ref:
            continue
        doc_text = row.get("semantic_text") or row.get("text") or ""
        if not doc_text.strip():
            continue
        lookup[ref] = {
            "title": row.get("title") or ref,
            "source": row.get("source") or "",
            "text": doc_text.strip(),
            "prepared_text": prepare_document_text(profile_key, doc_text.strip()),
        }
    return lookup


def build_pair_examples(rows: list[dict], corpus_lookup: dict, profile_key: str, split: str):
    examples = []
    for row in rows:
        if row.get("split") != split:
            continue
        positive = corpus_lookup.get(row.get("positive_ref"))
        if not positive:
            continue
        query = prepare_query_text(profile_key, row["query"])
        examples.append(InputExample(texts=[query, positive["prepared_text"]]))
    return examples


def build_triplet_examples(rows: list[dict], corpus_lookup: dict, profile_key: str, split: str):
    examples = []
    for row in rows:
        if row.get("split") != split:
            continue
        positive = corpus_lookup.get(row.get("positive_ref"))
        negative = corpus_lookup.get(row.get("negative_ref"))
        if not positive or not negative:
            continue
        query = prepare_query_text(profile_key, row["query"])
        examples.append(
            InputExample(texts=[query, positive["prepared_text"], negative["prepared_text"]])
        )
    return examples


def group_relevant_pairs(rows: list[dict], split: str):
    grouped = {}
    for row in rows:
        if row.get("split") != split:
            continue
        query = row["query"]
        entry = grouped.setdefault(
            query,
            {
                "query": query,
                "relevant_refs": set(),
                "language": row.get("language") or "",
                "benchmark": row.get("benchmark") or "",
            },
        )
        entry["relevant_refs"].add(row["positive_ref"])
    return grouped


def reciprocal_rank(result_refs: list[str], relevant_refs: set[str]):
    for rank, ref in enumerate(result_refs, start=1):
        if ref in relevant_refs:
            return 1.0 / rank
    return 0.0


def hit_rate(result_refs: list[str], relevant_refs: set[str]):
    return 1.0 if any(ref in relevant_refs for ref in result_refs) else 0.0


def ndcg_at_k(result_refs: list[str], relevant_refs: set[str], top_k: int):
    if not relevant_refs:
        return 0.0

    def dcg(refs):
        score = 0.0
        for rank, ref in enumerate(refs[:top_k], start=1):
            if ref in relevant_refs:
                score += 1.0 / np.log2(rank + 1)
        return score

    ideal_hits = min(len(relevant_refs), top_k)
    if ideal_hits == 0:
        return 0.0
    ideal_refs = list(relevant_refs)[:ideal_hits]
    ideal = dcg(ideal_refs)
    if ideal == 0:
        return 0.0
    return dcg(result_refs) / ideal


def evaluate_retriever(model, corpus_lookup: dict, pair_rows: list[dict], profile_key: str, top_k: int):
    dev_queries = group_relevant_pairs(pair_rows, "dev")
    if not dev_queries:
        return {
            "queries": 0,
            "mean_hit_rate_at_k": 0.0,
            "mean_mrr": 0.0,
            "mean_ndcg_at_k": 0.0,
            "language_breakdown": {},
        }

    doc_refs = list(corpus_lookup)
    doc_texts = [corpus_lookup[ref]["prepared_text"] for ref in doc_refs]
    query_texts = [
        prepare_query_text(profile_key, payload["query"])
        for payload in dev_queries.values()
    ]

    doc_embeddings = model.encode(
        doc_texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    query_embeddings = model.encode(
        query_texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    scores = np.matmul(query_embeddings, doc_embeddings.T)
    metrics = []
    language_scores = {}

    for row_index, payload in enumerate(dev_queries.values()):
        ranked_indices = np.argsort(scores[row_index])[::-1][:top_k]
        result_refs = [doc_refs[idx] for idx in ranked_indices]
        relevant_refs = payload["relevant_refs"]
        row_metrics = {
            "query": payload["query"],
            "language": payload["language"] or "unknown",
            "hit_rate_at_k": hit_rate(result_refs, relevant_refs),
            "mrr": reciprocal_rank(result_refs, relevant_refs),
            "ndcg_at_k": ndcg_at_k(result_refs, relevant_refs, top_k),
        }
        metrics.append(row_metrics)
        language_scores.setdefault(row_metrics["language"], []).append(row_metrics)

    return {
        "queries": len(metrics),
        "mean_hit_rate_at_k": mean(row["hit_rate_at_k"] for row in metrics),
        "mean_mrr": mean(row["mrr"] for row in metrics),
        "mean_ndcg_at_k": mean(row["ndcg_at_k"] for row in metrics),
        "language_breakdown": {
            language: {
                "queries": len(rows),
                "mean_hit_rate_at_k": mean(item["hit_rate_at_k"] for item in rows),
                "mean_mrr": mean(item["mrr"] for item in rows),
                "mean_ndcg_at_k": mean(item["ndcg_at_k"] for item in rows),
            }
            for language, rows in sorted(language_scores.items())
        },
    }


def warmup_steps(example_count: int, batch_size: int, epochs: int, ratio: float):
    if example_count <= 0 or epochs <= 0:
        return 0
    steps_per_epoch = max(1, example_count // batch_size)
    return max(1, int(steps_per_epoch * epochs * ratio))


def train_phase(
    model,
    examples,
    loss_obj,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    warmup_ratio: float,
):
    if not examples or epochs <= 0:
        return

    loader = DataLoader(
        examples,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=model.smart_batching_collate,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = max(1, len(loader) * epochs)
    num_warmup_steps = min(
        total_steps - 1,
        warmup_steps(len(examples), batch_size, epochs, warmup_ratio),
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps,
    )

    model.train()
    loss_obj.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for sentence_features, labels in loader:
            optimizer.zero_grad()
            loss_value = loss_obj(sentence_features, labels)
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += float(loss_value.detach().cpu())

        mean_loss = epoch_loss / max(1, len(loader))
        print(f"[TRAIN] epoch={epoch + 1}/{epochs} mean_loss={mean_loss:.4f}")


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_report(path: Path, summary: dict):
    lines = [
        "# Retriever Fine-Tuning Report",
        "",
        "Purpose:",
        "Fine-tune the English-centered MiniLM retriever with domain pairs and hard negatives.",
        "",
        "## Configuration",
        "",
        f"- Base profile: `{summary['base_profile']}`",
        f"- Base model: `{summary['base_model']}`",
        f"- Pair epochs: `{summary['pair_epochs']}`",
        f"- Triplet epochs: `{summary['triplet_epochs']}`",
        f"- Batch size: `{summary['batch_size']}`",
        f"- Learning rate: `{summary['learning_rate']}`",
        f"- Triplet margin: `{summary['triplet_margin']}`",
        "",
        "## Data",
        "",
        f"- Train pairs: `{summary['train_pairs']}`",
        f"- Dev pairs: `{summary['dev_pairs']}`",
        f"- Train triplets: `{summary['train_triplets']}`",
        f"- Dev triplets: `{summary['dev_triplets']}`",
        f"- Corpus docs: `{summary['corpus_docs']}`",
        "",
        "## Dev Retrieval",
        "",
        f"- Baseline nDCG@{summary['eval_top_k']}: `{summary['baseline_dev']['mean_ndcg_at_k']:.3f}`",
        f"- Fine-tuned nDCG@{summary['eval_top_k']}: `{summary['final_dev']['mean_ndcg_at_k']:.3f}`",
        f"- Baseline MRR: `{summary['baseline_dev']['mean_mrr']:.3f}`",
        f"- Fine-tuned MRR: `{summary['final_dev']['mean_mrr']:.3f}`",
        f"- Baseline Hit@{summary['eval_top_k']}: `{summary['baseline_dev']['mean_hit_rate_at_k']:.3f}`",
        f"- Fine-tuned Hit@{summary['eval_top_k']}: `{summary['final_dev']['mean_hit_rate_at_k']:.3f}`",
        "",
        "## Outputs",
        "",
        f"- Final model: `{summary['final_model_dir']}`",
        f"- Summary: `{summary['summary_path']}`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    ensure_model_cache_dirs()

    profile = get_profile(args.base_profile)
    corpus_rows = load_jsonl(args.corpus)
    pair_rows = load_jsonl(args.pairs)
    triplet_rows = load_jsonl(args.triplets)
    corpus_lookup = build_corpus_lookup(corpus_rows, profile.key)

    train_pair_examples = build_pair_examples(pair_rows, corpus_lookup, profile.key, "train")
    train_triplet_examples = build_triplet_examples(
        triplet_rows,
        corpus_lookup,
        profile.key,
        "train",
    )

    model = SentenceTransformer(profile.model_name)
    baseline_dev = evaluate_retriever(
        model,
        corpus_lookup,
        pair_rows,
        profile.key,
        args.eval_top_k,
    )

    if not args.skip_pair_phase:
        pair_loss = losses.MultipleNegativesRankingLoss(model)
        train_phase(
            model,
            train_pair_examples,
            pair_loss,
            args.pair_epochs,
            args.batch_size,
            args.learning_rate,
            args.warmup_ratio,
        )

    after_pair_dev = evaluate_retriever(
        model,
        corpus_lookup,
        pair_rows,
        profile.key,
        args.eval_top_k,
    )

    if not args.skip_triplet_phase:
        triplet_loss = losses.TripletLoss(
            model,
            distance_metric=losses.TripletDistanceMetric.COSINE,
            triplet_margin=args.triplet_margin,
        )
        train_phase(
            model,
            train_triplet_examples,
            triplet_loss,
            args.triplet_epochs,
            args.batch_size,
            args.learning_rate,
            args.warmup_ratio,
        )

    final_dev = evaluate_retriever(
        model,
        corpus_lookup,
        pair_rows,
        profile.key,
        args.eval_top_k,
    )

    final_model_dir = args.output_dir / "final"
    final_model_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(final_model_dir))

    summary = {
        "base_profile": profile.key,
        "base_model": profile.model_name,
        "pair_epochs": args.pair_epochs,
        "triplet_epochs": args.triplet_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "triplet_margin": args.triplet_margin,
        "eval_top_k": args.eval_top_k,
        "train_pairs": len(train_pair_examples),
        "dev_pairs": sum(1 for row in pair_rows if row.get("split") == "dev"),
        "train_triplets": len(train_triplet_examples),
        "dev_triplets": sum(1 for row in triplet_rows if row.get("split") == "dev"),
        "corpus_docs": len(corpus_lookup),
        "baseline_dev": baseline_dev,
        "after_pair_dev": after_pair_dev,
        "final_dev": final_dev,
        "final_model_dir": str(final_model_dir),
    }

    summary_path = args.output_dir / "training_summary.json"
    summary["summary_path"] = str(summary_path)
    report_path = REPORT_DIR / "retriever_finetune_report.md"
    write_json(summary_path, summary)
    write_report(report_path, summary)

    print(f"[OK] Saved fine-tuned model to {final_model_dir}")
    print(f"[OK] Wrote training summary to {summary_path}")
    print(f"[OK] Wrote training report to {report_path}")
    print(
        "[DEV] nDCG@{k}: {before:.3f} -> {after:.3f}".format(
            k=args.eval_top_k,
            before=baseline_dev["mean_ndcg_at_k"],
            after=final_dev["mean_ndcg_at_k"],
        )
    )


if __name__ == "__main__":
    main()
