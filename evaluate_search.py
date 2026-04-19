import argparse
import csv
import json
import math
import sys
from pathlib import Path
from statistics import mean

from bm25 import BM25Index, search as bm25_search
from hybrid import DEFAULT_CANDIDATE_K, DEFAULT_SEMANTIC_WEIGHT, search as hybrid_search
from search import load_mappings, load_search_engine, search as semantic_search

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_JUDGMENTS = BASE_DIR / "data" / "evaluation" / "relevance_judgments.json"
DEFAULT_REPORT_DIR = BASE_DIR / "reports" / "evaluation"
DETAILS_CSV = "evaluation_details.csv"
SUMMARY_CSV = "evaluation_summary.csv"
SUMMARY_JSON = "evaluation_summary.json"


def configure_output():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate semantic, BM25, and hybrid dataset search with relevance judgments."
    )
    parser.add_argument("--judgments", type=Path, default=DEFAULT_JUDGMENTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--methods",
        default="semantic,bm25,hybrid",
        help="Comma-separated methods: semantic,bm25,hybrid",
    )
    parser.add_argument("--hybrid-alpha", type=float, default=DEFAULT_SEMANTIC_WEIGHT)
    parser.add_argument("--hybrid-candidates", type=int, default=DEFAULT_CANDIDATE_K)
    return parser.parse_args()


def load_judgments(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        judgments = json.load(f)
    if not isinstance(judgments, list):
        raise ValueError("Evaluation judgments must be a JSON list.")
    return judgments


def validate_judgments(judgments, mappings):
    known_refs = {item.get("ref") for item in mappings.values()}
    missing = []

    for judgment in judgments:
        relevant_refs = judgment.get("relevant_refs") or []
        if not relevant_refs:
            missing.append((judgment.get("id"), "<empty relevant_refs>"))
            continue
        for ref in relevant_refs:
            if ref not in known_refs:
                missing.append((judgment.get("id"), ref))

    if missing:
        lines = [f"{query_id}: {ref}" for query_id, ref in missing]
        raise ValueError(
            "Some relevance references are not present in mappings.json:\n"
            + "\n".join(lines)
        )


def apply_source_filter(results, source_filter):
    if source_filter == "all":
        return results
    return [
        (score, item)
        for score, item in results
        if (item.get("source") or "").lower() == source_filter
    ]


def run_method(method, query, top_k, source_filter, context):
    candidate_k = len(context["mappings"]) if source_filter != "all" else top_k

    if method == "semantic":
        if context.get("semantic_engine") is None:
            context["semantic_engine"] = load_search_engine()
        model, index, mappings = context["semantic_engine"]
        if source_filter != "all":
            candidate_k = index.ntotal
        raw_results = semantic_search(query, model, index, mappings, top_k=candidate_k)
    elif method == "bm25":
        if context.get("bm25_index") is None:
            context["bm25_index"] = BM25Index(context["mappings"])
        if source_filter != "all":
            candidate_k = len(context["bm25_index"].doc_ids)
        raw_results = bm25_search(query, context["bm25_index"], top_k=candidate_k)
    elif method == "hybrid":
        if context.get("semantic_engine") is None:
            context["semantic_engine"] = load_search_engine()
        if context.get("bm25_index") is None:
            context["bm25_index"] = BM25Index(context["mappings"])
        model, index, mappings = context["semantic_engine"]
        candidate_k = max(top_k, min(context["hybrid_candidates"], len(mappings)))
        if source_filter != "all":
            candidate_k = len(mappings)
        raw_results = hybrid_search(
            query,
            model,
            index,
            mappings,
            context["bm25_index"],
            top_k=candidate_k if source_filter != "all" else top_k,
            semantic_weight=context["hybrid_alpha"],
            candidate_k=candidate_k,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    return apply_source_filter(raw_results, source_filter)[:top_k]


def precision_at_k(result_refs, relevant_refs, top_k):
    if top_k <= 0:
        return 0.0
    hits = sum(1 for ref in result_refs[:top_k] if ref in relevant_refs)
    return hits / top_k


def recall_at_k(result_refs, relevant_refs, top_k):
    if not relevant_refs:
        return 0.0
    hits = sum(1 for ref in result_refs[:top_k] if ref in relevant_refs)
    return hits / len(relevant_refs)


def reciprocal_rank(result_refs, relevant_refs):
    for rank, ref in enumerate(result_refs, start=1):
        if ref in relevant_refs:
            return 1 / rank
    return 0.0


def dcg_at_k(result_refs, relevant_refs, top_k):
    score = 0.0
    for rank, ref in enumerate(result_refs[:top_k], start=1):
        relevance = 1 if ref in relevant_refs else 0
        if relevance:
            score += relevance / math.log2(rank + 1)
    return score


def ndcg_at_k(result_refs, relevant_refs, top_k):
    ideal_hits = min(len(relevant_refs), top_k)
    if ideal_hits == 0:
        return 0.0
    ideal_refs = list(relevant_refs)[:ideal_hits]
    ideal_dcg = dcg_at_k(ideal_refs, relevant_refs, top_k)
    if ideal_dcg == 0:
        return 0.0
    return dcg_at_k(result_refs, relevant_refs, top_k) / ideal_dcg


def evaluate(judgments, methods, top_k, context):
    rows = []

    for judgment in judgments:
        query = judgment["query"]
        source_filter = (judgment.get("source_filter") or "all").lower()
        relevant_refs = set(judgment.get("relevant_refs") or [])

        for method in methods:
            results = run_method(method, query, top_k, source_filter, context)
            result_refs = [item.get("ref") for _, item in results]
            hits = [ref for ref in result_refs if ref in relevant_refs]

            rows.append(
                {
                    "query_id": judgment.get("id"),
                    "query": query,
                    "category": judgment.get("category"),
                    "method": method,
                    "source_filter": source_filter,
                    "top_k": top_k,
                    "relevant_count": len(relevant_refs),
                    "hits": len(hits),
                    "precision_at_k": precision_at_k(result_refs, relevant_refs, top_k),
                    "recall_at_k": recall_at_k(result_refs, relevant_refs, top_k),
                    "mrr": reciprocal_rank(result_refs, relevant_refs),
                    "ndcg_at_k": ndcg_at_k(result_refs, relevant_refs, top_k),
                    "retrieved_refs": " | ".join(ref or "" for ref in result_refs),
                    "hit_refs": " | ".join(hits),
                }
            )

    return rows


def summarize(rows):
    methods = sorted({row["method"] for row in rows})
    summary = []

    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        summary.append(
            {
                "method": method,
                "queries": len(method_rows),
                "mean_precision_at_k": mean(row["precision_at_k"] for row in method_rows),
                "mean_recall_at_k": mean(row["recall_at_k"] for row in method_rows),
                "mean_mrr": mean(row["mrr"] for row in method_rows),
                "mean_ndcg_at_k": mean(row["ndcg_at_k"] for row in method_rows),
            }
        )

    return summary


def write_csv(path: Path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    configure_output()
    args = parse_args()

    methods = [method.strip().lower() for method in args.methods.split(",") if method.strip()]
    unknown_methods = set(methods) - {"semantic", "bm25", "hybrid"}
    if unknown_methods:
        raise ValueError(f"Unknown methods: {', '.join(sorted(unknown_methods))}")

    top_k = max(1, args.top_k)
    judgments = load_judgments(args.judgments)
    mappings = load_mappings()
    validate_judgments(judgments, mappings)

    context = {
        "mappings": mappings,
        "hybrid_alpha": args.hybrid_alpha,
        "hybrid_candidates": args.hybrid_candidates,
    }
    rows = evaluate(judgments, methods, top_k, context)
    summary = summarize(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / DETAILS_CSV, rows)
    write_csv(args.out_dir / SUMMARY_CSV, summary)
    with (args.out_dir / SUMMARY_JSON).open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[OK] Evaluated {len(judgments)} queries at top_k={top_k}")
    for row in summary:
        print(
            f"  {row['method']}: "
            f"P@{top_k}={row['mean_precision_at_k']:.3f} "
            f"R@{top_k}={row['mean_recall_at_k']:.3f} "
            f"MRR={row['mean_mrr']:.3f} "
            f"nDCG@{top_k}={row['mean_ndcg_at_k']:.3f}"
        )
    print(f"  details={args.out_dir / DETAILS_CSV}")
    print(f"  summary={args.out_dir / SUMMARY_CSV}")


if __name__ == "__main__":
    main()
