import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from statistics import mean, median

from bm25 import BM25Index, search as bm25_search
from hybrid import DEFAULT_CANDIDATE_K, DEFAULT_SEMANTIC_WEIGHT, search as hybrid_search
from reranker import DEFAULT_RERANK_DEPTH
from search import load_mappings, load_search_engine, search as semantic_search
from search_profiles import DEFAULT_PROFILE_KEY, get_profile

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_JUDGMENTS = BASE_DIR / "data" / "evaluation" / "relevance_judgments.json"
DEFAULT_REPORT_DIR = BASE_DIR / "reports" / "evaluation"
DETAILS_CSV = "evaluation_details.csv"
SUMMARY_CSV = "evaluation_summary.csv"
SUMMARY_JSON = "evaluation_summary.json"
CATEGORY_CSV = "evaluation_by_category.csv"
CATEGORY_JSON = "evaluation_by_category.json"
PAIRWISE_JSON = "evaluation_pairwise.json"
OPTIONAL_GROUP_FIELDS = (
    "benchmark",
    "query_style",
    "language",
    "study_slice",
    "topic",
    "direction",
    "anchor_ref",
    "anchor_title",
    "anchor_source",
    "target_source",
)

METRIC_FIELDS = (
    "precision_at_1",
    "precision_at_k",
    "recall_at_k",
    "hit_rate_at_k",
    "mrr",
    "ap_at_k",
    "r_precision",
    "ndcg_at_k",
)


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
    parser.add_argument("--disable-rerank", action="store_true")
    parser.add_argument("--disable-quality-penalty", action="store_true")
    parser.add_argument("--disable-tr-fusion", action="store_true")
    parser.add_argument("--rerank-depth", type=int, default=DEFAULT_RERANK_DEPTH)
    parser.add_argument("--profile", default=DEFAULT_PROFILE_KEY)
    return parser.parse_args()


def parse_profiles(profile_arg):
    profile_keys = [part.strip() for part in str(profile_arg).split(",") if part.strip()]
    if not profile_keys:
        profile_keys = [DEFAULT_PROFILE_KEY]
    return [get_profile(profile_key) for profile_key in profile_keys]


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
            context["semantic_engine"] = load_search_engine(context["profile"])
        model, index, mappings = context["semantic_engine"]
        if source_filter != "all":
            candidate_k = index.ntotal
        raw_results = semantic_search(
            query,
            model,
            index,
            mappings,
            top_k=candidate_k,
            profile_key=context["profile"],
            enable_rerank=context.get("enable_rerank", True),
            rerank_depth=context.get("rerank_depth", DEFAULT_RERANK_DEPTH),
            enable_quality_penalty=context.get("enable_quality_penalty", True),
            enable_tr_fusion=context.get("enable_tr_fusion", True),
        )
    elif method == "bm25":
        if context.get("bm25_index") is None:
            context["bm25_index"] = BM25Index(context["mappings"])
        if source_filter != "all":
            candidate_k = len(context["bm25_index"].doc_ids)
        raw_results = bm25_search(query, context["bm25_index"], top_k=candidate_k)
    elif method == "hybrid":
        if context.get("semantic_engine") is None:
            context["semantic_engine"] = load_search_engine(context["profile"])
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
            profile_key=context["profile"],
            enable_rerank=context.get("enable_rerank", True),
            rerank_depth=context.get("rerank_depth", DEFAULT_RERANK_DEPTH),
            enable_quality_penalty=context.get("enable_quality_penalty", True),
            enable_tr_fusion=context.get("enable_tr_fusion", True),
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    return apply_source_filter(raw_results, source_filter)[:top_k]


def warm_context(methods, context):
    methods = {method.strip().lower() for method in methods}

    if "semantic" in methods or "hybrid" in methods:
        if context.get("semantic_engine") is None:
            context["semantic_engine"] = load_search_engine(context["profile"])

    if "bm25" in methods or "hybrid" in methods:
        if context.get("bm25_index") is None:
            context["bm25_index"] = BM25Index(context["mappings"])


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


def first_relevant_rank(result_refs, relevant_refs):
    for rank, ref in enumerate(result_refs, start=1):
        if ref in relevant_refs:
            return rank
    return None


def reciprocal_rank(result_refs, relevant_refs):
    rank = first_relevant_rank(result_refs, relevant_refs)
    if rank is None:
        return 0.0
    return 1 / rank


def hit_rate_at_k(result_refs, relevant_refs, top_k):
    return 1.0 if any(ref in relevant_refs for ref in result_refs[:top_k]) else 0.0


def average_precision_at_k(result_refs, relevant_refs, top_k):
    if not relevant_refs:
        return 0.0

    hits = 0
    precision_sum = 0.0
    for rank, ref in enumerate(result_refs[:top_k], start=1):
        if ref in relevant_refs:
            hits += 1
            precision_sum += hits / rank

    normalization = min(len(relevant_refs), top_k)
    if normalization == 0:
        return 0.0
    return precision_sum / normalization


def r_precision(result_refs, relevant_refs):
    if not relevant_refs:
        return 0.0
    r = len(relevant_refs)
    hits = sum(1 for ref in result_refs[:r] if ref in relevant_refs)
    return hits / r


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


def build_row(judgment, method, profile, top_k, result_refs, latency_ms):
    relevant_refs = set(judgment.get("relevant_refs") or [])
    hits = [ref for ref in result_refs[:top_k] if ref in relevant_refs]
    rank = first_relevant_rank(result_refs, relevant_refs)

    row = {
        "query_id": judgment.get("id"),
        "query": judgment["query"],
        "category": judgment.get("category"),
        "method": method,
        "profile": profile,
        "source_filter": (judgment.get("source_filter") or "all").lower(),
        "top_k": top_k,
        "relevant_count": len(relevant_refs),
        "hits": len(hits),
        "latency_ms": latency_ms,
        "first_relevant_rank": rank if rank is not None else "",
        "precision_at_1": precision_at_k(result_refs, relevant_refs, 1),
        "precision_at_k": precision_at_k(result_refs, relevant_refs, top_k),
        "recall_at_k": recall_at_k(result_refs, relevant_refs, top_k),
        "hit_rate_at_k": hit_rate_at_k(result_refs, relevant_refs, top_k),
        "mrr": reciprocal_rank(result_refs, relevant_refs),
        "ap_at_k": average_precision_at_k(result_refs, relevant_refs, top_k),
        "r_precision": r_precision(result_refs, relevant_refs),
        "ndcg_at_k": ndcg_at_k(result_refs, relevant_refs, top_k),
        "retrieved_refs": " | ".join(ref or "" for ref in result_refs[:top_k]),
        "hit_refs": " | ".join(hits),
    }
    for field in OPTIONAL_GROUP_FIELDS:
        if field in judgment:
            row[field] = judgment.get(field)
    return row


def evaluate(judgments, methods, top_k, context):
    rows = []

    for judgment in judgments:
        query = judgment["query"]
        source_filter = (judgment.get("source_filter") or "all").lower()
        relevant_refs = set(judgment.get("relevant_refs") or [])
        eval_depth = max(top_k, len(relevant_refs))

        for method in methods:
            started_at = time.perf_counter()
            results = run_method(method, query, eval_depth, source_filter, context)
            latency_ms = (time.perf_counter() - started_at) * 1000.0
            result_refs = [item.get("ref") for _, item in results]
            rows.append(
                build_row(
                    judgment,
                    method,
                    context["profile"],
                    top_k,
                    result_refs,
                    latency_ms,
                )
            )

    return rows


def summarize_group(rows, group_key=None):
    groups = {}
    for row in rows:
        value = row.get(group_key) if group_key else row["method"]
        if not value:
            value = "uncategorized"
        groups.setdefault(value, []).append(row)

    summary = []
    for group_value in sorted(groups):
        group_rows = groups[group_value]
        systems = sorted({(row["profile"], row["method"]) for row in group_rows})
        for profile, method in systems:
            method_rows = [
                row
                for row in group_rows
                if row["method"] == method and row["profile"] == profile
            ]
            ranks = [
                int(row["first_relevant_rank"])
                for row in method_rows
                if row["first_relevant_rank"] != ""
            ]
            record = {
                "method": method,
                "profile": method_rows[0]["profile"] if method_rows else "",
                "queries": len(method_rows),
                "queries_with_hit": sum(1 for row in method_rows if row["hit_rate_at_k"] > 0),
                "queries_top1_hit": sum(1 for row in method_rows if row["precision_at_1"] > 0),
                "mean_latency_ms": mean(row["latency_ms"] for row in method_rows),
                "median_latency_ms": median(row["latency_ms"] for row in method_rows),
                "mean_precision_at_1": mean(row["precision_at_1"] for row in method_rows),
                "mean_precision_at_k": mean(row["precision_at_k"] for row in method_rows),
                "mean_recall_at_k": mean(row["recall_at_k"] for row in method_rows),
                "mean_hit_rate_at_k": mean(row["hit_rate_at_k"] for row in method_rows),
                "mean_mrr": mean(row["mrr"] for row in method_rows),
                "mean_ap_at_k": mean(row["ap_at_k"] for row in method_rows),
                "mean_r_precision": mean(row["r_precision"] for row in method_rows),
                "mean_ndcg_at_k": mean(row["ndcg_at_k"] for row in method_rows),
                "mean_first_relevant_rank": mean(ranks) if ranks else "",
                "median_first_relevant_rank": median(ranks) if ranks else "",
            }
            if group_key:
                record[group_key] = group_value
            summary.append(record)

    return summary


def summarize(rows):
    return summarize_group(rows)


def summarize_by_category(rows):
    categories = [row for row in rows if row.get("category")]
    return summarize_group(categories, group_key="category")


def summarize_by_field(rows, field_name):
    scoped = [row for row in rows if row.get(field_name)]
    return summarize_group(scoped, group_key=field_name)


def build_pairwise_comparison(rows):
    rows_by_query = {}
    for row in rows:
        system_key = f"{row['profile']}::{row['method']}"
        rows_by_query.setdefault(row["query_id"], {})[system_key] = row

    systems = sorted({f"{row['profile']}::{row['method']}" for row in rows})
    summary = []
    epsilon = 1e-12

    for i, left in enumerate(systems):
        for right in systems[i + 1 :]:
            metric_comparisons = {}
            for metric in METRIC_FIELDS:
                wins_left = 0
                wins_right = 0
                ties = 0
                deltas = []

                for query_id, method_rows in rows_by_query.items():
                    left_row = method_rows.get(left)
                    right_row = method_rows.get(right)
                    if left_row is None or right_row is None:
                        continue
                    delta = left_row[metric] - right_row[metric]
                    deltas.append(delta)
                    if delta > epsilon:
                        wins_left += 1
                    elif delta < -epsilon:
                        wins_right += 1
                    else:
                        ties += 1

                metric_comparisons[metric] = {
                    "left_wins": wins_left,
                    "right_wins": wins_right,
                    "ties": ties,
                    "mean_delta": mean(deltas) if deltas else 0.0,
                }

            summary.append(
                {
                    "left_profile": left.split("::", 1)[0],
                    "left_method": left.split("::", 1)[1],
                    "right_profile": right.split("::", 1)[0],
                    "right_method": right.split("::", 1)[1],
                    "queries": len(rows_by_query),
                    "metrics": metric_comparisons,
                }
            )

    return summary


def write_csv(path: Path, rows):
    if not rows:
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
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
    profiles = parse_profiles(args.profile)
    judgments = load_judgments(args.judgments)
    rows = []
    for profile in profiles:
        mappings = load_mappings(profile.key)
        validate_judgments(judgments, mappings)
        context = {
            "mappings": mappings,
            "hybrid_alpha": args.hybrid_alpha,
            "hybrid_candidates": args.hybrid_candidates,
            "profile": profile.key,
            "enable_rerank": not args.disable_rerank,
            "enable_quality_penalty": not args.disable_quality_penalty,
            "enable_tr_fusion": not args.disable_tr_fusion,
            "rerank_depth": args.rerank_depth,
        }
        warm_context(methods, context)
        rows.extend(evaluate(judgments, methods, top_k, context))
    summary = summarize(rows)
    category_summary = summarize_by_category(rows)
    benchmark_summary = summarize_by_field(rows, "benchmark")
    query_style_summary = summarize_by_field(rows, "query_style")
    pairwise = build_pairwise_comparison(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / DETAILS_CSV, rows)
    write_csv(args.out_dir / SUMMARY_CSV, summary)
    with (args.out_dir / SUMMARY_JSON).open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    if category_summary:
        write_csv(args.out_dir / CATEGORY_CSV, category_summary)
        with (args.out_dir / CATEGORY_JSON).open("w", encoding="utf-8") as f:
            json.dump(category_summary, f, ensure_ascii=False, indent=2)
    if benchmark_summary:
        write_csv(args.out_dir / "evaluation_by_benchmark.csv", benchmark_summary)
        with (args.out_dir / "evaluation_by_benchmark.json").open("w", encoding="utf-8") as f:
            json.dump(benchmark_summary, f, ensure_ascii=False, indent=2)
    if query_style_summary:
        write_csv(args.out_dir / "evaluation_by_query_style.csv", query_style_summary)
        with (args.out_dir / "evaluation_by_query_style.json").open("w", encoding="utf-8") as f:
            json.dump(query_style_summary, f, ensure_ascii=False, indent=2)
    with (args.out_dir / PAIRWISE_JSON).open("w", encoding="utf-8") as f:
        json.dump(pairwise, f, ensure_ascii=False, indent=2)

    profile_label = ", ".join(profile.key for profile in profiles)
    print(f"[OK] Evaluated {len(judgments)} queries at top_k={top_k} with profiles={profile_label}")
    for row in summary:
        print(
            f"  {row['profile']} / {row['method']}: "
            f"S@1={row['mean_precision_at_1']:.3f} "
            f"P@{top_k}={row['mean_precision_at_k']:.3f} "
            f"R@{top_k}={row['mean_recall_at_k']:.3f} "
            f"Hit@{top_k}={row['mean_hit_rate_at_k']:.3f} "
            f"MRR={row['mean_mrr']:.3f} "
            f"MAP@{top_k}={row['mean_ap_at_k']:.3f} "
            f"R-Prec={row['mean_r_precision']:.3f} "
            f"nDCG@{top_k}={row['mean_ndcg_at_k']:.3f}"
        )
    print(f"  details={args.out_dir / DETAILS_CSV}")
    print(f"  summary={args.out_dir / SUMMARY_CSV}")
    if category_summary:
        print(f"  by_category={args.out_dir / CATEGORY_CSV}")
    if benchmark_summary:
        print(f"  by_benchmark={args.out_dir / 'evaluation_by_benchmark.csv'}")
    if query_style_summary:
        print(f"  by_query_style={args.out_dir / 'evaluation_by_query_style.csv'}")
    print(f"  pairwise={args.out_dir / PAIRWISE_JSON}")


if __name__ == "__main__":
    main()
