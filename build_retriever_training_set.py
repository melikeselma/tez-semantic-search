import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from evaluate_search import load_judgments, run_method, validate_judgments, warm_context
from query_understanding import build_query_plan
from search import load_mappings
from search_profiles import DEFAULT_PROFILE_KEY

BASE_DIR = Path(__file__).resolve().parent
EVAL_DIR = BASE_DIR / "data" / "evaluation"
TRAIN_DIR = BASE_DIR / "data" / "training"
REPORT_DIR = BASE_DIR / "reports" / "training"

JUDGMENT_SOURCES = [
    EVAL_DIR / "relevance_judgments_rq1.json",
    EVAL_DIR / "relevance_judgments_rq2.json",
    EVAL_DIR / "relevance_judgments_tr.json",
    EVAL_DIR / "relevance_judgments_semantic_intent.json",
]

PROFILE_KEY = DEFAULT_PROFILE_KEY
MINING_METHODS = ("semantic", "hybrid", "bm25")
TOP_K_PER_METHOD = 40
NEGATIVES_PER_POSITIVE = 3
DEV_SPLIT_MOD = 7
DEV_SPLIT_THRESHOLD = 1

CORPUS_PATH = TRAIN_DIR / "retriever_corpus.jsonl"
PAIRS_PATH = TRAIN_DIR / "retriever_pairs.jsonl"
TRIPLETS_PATH = TRAIN_DIR / "retriever_triplets.jsonl"
SUMMARY_PATH = TRAIN_DIR / "retriever_training_summary.json"
REPORT_PATH = REPORT_DIR / "retriever_training_set_report.md"


def stable_split(key: str) -> str:
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % DEV_SPLIT_MOD
    return "dev" if bucket < DEV_SPLIT_THRESHOLD else "train"


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_all_judgments():
    merged = []
    for path in JUDGMENT_SOURCES:
        records = load_judgments(path)
        for record in records:
            enriched = dict(record)
            enriched.setdefault("benchmark", path.stem.replace("relevance_judgments_", ""))
            query_plan = build_query_plan(enriched["query"])
            detected_language = (query_plan.get("detected_language") or "").lower()
            if detected_language == "unknown":
                detected_language = ""
            enriched["language"] = (enriched.get("language") or detected_language or "").lower()
            enriched["_query_plan"] = query_plan
            merged.append(enriched)
    return merged


def build_ref_index(mappings: dict):
    return {item.get("ref"): item for item in mappings.values() if item.get("ref")}


def collect_candidate_pool(judgment: dict, context: dict):
    query = judgment["query"]
    source_filter = (judgment.get("source_filter") or "all").lower()
    pool = {}

    for method in MINING_METHODS:
        results = run_method(method, query, TOP_K_PER_METHOD, source_filter, context)
        for rank, (score, item) in enumerate(results, start=1):
            ref = item.get("ref")
            if not ref:
                continue
            entry = pool.setdefault(
                ref,
                {
                    "item": item,
                    "methods": {},
                },
            )
            entry["methods"][method] = {
                "rank": rank,
                "score": float(score),
            }

    return pool


def overlap_size(left, right) -> int:
    return len(set(left or []) & set(right or []))


def candidate_hardness(
    query_domains,
    positive_item: dict,
    candidate_entry: dict,
):
    item = candidate_entry["item"]
    methods = candidate_entry["methods"]
    score = 0.0

    for method, payload in methods.items():
        rank = payload["rank"]
        if method == "semantic":
            score += 4.0 / rank
        elif method == "hybrid":
            score += 3.0 / rank
        else:
            score += 1.5 / rank

    score += 0.8 * overlap_size(query_domains, item.get("inferred_domains") or [])
    score += 1.2 * overlap_size(
        positive_item.get("inferred_domains") or [],
        item.get("inferred_domains") or [],
    )
    score += 0.7 * overlap_size(
        positive_item.get("inferred_use_cases") or [],
        item.get("inferred_use_cases") or [],
    )
    score += 0.4 * overlap_size(
        positive_item.get("inferred_modalities") or [],
        item.get("inferred_modalities") or [],
    )
    return score


def backfill_negatives(
    mappings: dict,
    query_domains,
    positive_item: dict,
    relevant_refs: set[str],
    source_filter: str,
    already_selected: set[str],
):
    candidates = []
    for item in mappings.values():
        ref = item.get("ref")
        if not ref or ref in relevant_refs or ref in already_selected:
            continue
        item_source = (item.get("source") or "").lower()
        if source_filter != "all" and item_source != source_filter:
            continue

        score = 0.0
        score += 0.8 * overlap_size(query_domains, item.get("inferred_domains") or [])
        score += 1.0 * overlap_size(
            positive_item.get("inferred_domains") or [],
            item.get("inferred_domains") or [],
        )
        score += 0.6 * overlap_size(
            positive_item.get("inferred_use_cases") or [],
            item.get("inferred_use_cases") or [],
        )
        score += 0.4 * overlap_size(
            positive_item.get("inferred_modalities") or [],
            item.get("inferred_modalities") or [],
        )
        if score <= 0:
            continue
        candidates.append((score, item))

    candidates.sort(key=lambda row: row[0], reverse=True)
    return candidates


def choose_negatives(judgment: dict, pool: dict, mappings: dict, ref_to_item: dict):
    relevant_refs = set(judgment.get("relevant_refs") or [])
    source_filter = (judgment.get("source_filter") or "all").lower()
    query_plan = judgment.get("_query_plan") or build_query_plan(judgment["query"])
    query_domains = query_plan.get("domains") or []
    triplets = []

    for positive_ref in judgment.get("relevant_refs") or []:
        positive_item = ref_to_item.get(positive_ref)
        if not positive_item:
            continue

        ranked = []
        for ref, candidate_entry in pool.items():
            if ref in relevant_refs:
                continue
            hardness = candidate_hardness(query_domains, positive_item, candidate_entry)
            ranked.append((hardness, ref, candidate_entry))

        ranked.sort(key=lambda row: row[0], reverse=True)

        selected_refs = set()
        selected = []
        for hardness, ref, candidate_entry in ranked:
            if len(selected) >= NEGATIVES_PER_POSITIVE:
                break
            if ref in selected_refs:
                continue
            selected_refs.add(ref)
            selected.append(
                {
                    "negative_ref": ref,
                    "negative_item": candidate_entry["item"],
                    "negative_hardness": round(hardness, 6),
                    "negative_strategy": "retrieval_confusion",
                    "mined_from_methods": sorted(candidate_entry["methods"]),
                }
            )

        if len(selected) < NEGATIVES_PER_POSITIVE:
            backfills = backfill_negatives(
                mappings,
                query_domains,
                positive_item,
                relevant_refs,
                source_filter,
                selected_refs,
            )
            for hardness, item in backfills:
                if len(selected) >= NEGATIVES_PER_POSITIVE:
                    break
                ref = item.get("ref")
                if not ref or ref in selected_refs:
                    continue
                selected_refs.add(ref)
                selected.append(
                    {
                        "negative_ref": ref,
                        "negative_item": item,
                        "negative_hardness": round(hardness, 6),
                        "negative_strategy": "domain_neighbor_backfill",
                        "mined_from_methods": [],
                    }
                )

        split = stable_split(judgment.get("id") or judgment["query"])
        for negative in selected:
            triplets.append(
                {
                    "query_id": judgment.get("id"),
                    "query": judgment["query"],
                    "split": split,
                    "benchmark": judgment.get("benchmark") or "unknown",
                    "category": judgment.get("category") or "",
                    "topic": judgment.get("topic") or "",
                    "language": judgment.get("language") or "",
                    "query_style": judgment.get("query_style") or "",
                    "study_slice": judgment.get("study_slice") or "",
                    "source_filter": source_filter,
                    "positive_ref": positive_ref,
                    "positive_title": positive_item.get("title") or positive_ref,
                    "positive_source": positive_item.get("source") or "",
                    "negative_ref": negative["negative_ref"],
                    "negative_title": negative["negative_item"].get("title") or negative["negative_ref"],
                    "negative_source": negative["negative_item"].get("source") or "",
                    "negative_strategy": negative["negative_strategy"],
                    "negative_hardness": negative["negative_hardness"],
                    "mined_from_methods": negative["mined_from_methods"],
                }
            )

    return triplets


def build_pair_rows(judgments: list[dict], ref_to_item: dict):
    rows = []
    seen = set()
    for judgment in judgments:
        split = stable_split(judgment.get("id") or judgment["query"])
        for positive_ref in judgment.get("relevant_refs") or []:
            positive_item = ref_to_item.get(positive_ref)
            if not positive_item:
                continue
            key = (judgment["query"], positive_ref)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "query_id": judgment.get("id"),
                    "query": judgment["query"],
                    "split": split,
                    "benchmark": judgment.get("benchmark") or "unknown",
                    "category": judgment.get("category") or "",
                    "topic": judgment.get("topic") or "",
                    "language": judgment.get("language") or "",
                    "query_style": judgment.get("query_style") or "",
                    "study_slice": judgment.get("study_slice") or "",
                    "source_filter": (judgment.get("source_filter") or "all").lower(),
                    "positive_ref": positive_ref,
                    "positive_title": positive_item.get("title") or positive_ref,
                    "positive_source": positive_item.get("source") or "",
                }
            )
    return rows


def build_corpus_rows(mappings: dict):
    rows = []
    for item in mappings.values():
        ref = item.get("ref")
        if not ref:
            continue
        rows.append(
            {
                "ref": ref,
                "title": item.get("title") or ref,
                "source": item.get("source") or "",
                "url": item.get("url") or "",
                "text": item.get("text") or "",
                "semantic_text": item.get("semantic_text") or item.get("text") or "",
                "semantic_summary": item.get("semantic_summary") or "",
                "keywords": item.get("keywords") or [],
                "language_hint": item.get("language_hint") or "",
                "inferred_domains": item.get("inferred_domains") or [],
                "inferred_use_cases": item.get("inferred_use_cases") or [],
                "inferred_modalities": item.get("inferred_modalities") or [],
                "quality_flags": item.get("quality_flags") or [],
            }
        )
    return rows


def summarize(judgments, pairs, triplets):
    benchmark_counts = Counter(item.get("benchmark") or "unknown" for item in judgments)
    language_counts = Counter(item.get("language") or "unknown" for item in judgments)
    pair_splits = Counter(item["split"] for item in pairs)
    triplet_splits = Counter(item["split"] for item in triplets)
    negative_strategies = Counter(item["negative_strategy"] for item in triplets)
    method_usage = Counter()
    for row in triplets:
        for method in row.get("mined_from_methods") or []:
            method_usage[method] += 1

    return {
        "profile": PROFILE_KEY,
        "judgment_sources": [str(path) for path in JUDGMENT_SOURCES],
        "queries": len(judgments),
        "unique_query_texts": len({item["query"] for item in judgments}),
        "pairs": len(pairs),
        "triplets": len(triplets),
        "unique_positive_refs": len({item["positive_ref"] for item in pairs}),
        "unique_negative_refs": len({item["negative_ref"] for item in triplets}),
        "benchmark_counts": dict(benchmark_counts),
        "language_counts": dict(language_counts),
        "pair_splits": dict(pair_splits),
        "triplet_splits": dict(triplet_splits),
        "negative_strategies": dict(negative_strategies),
        "method_usage": dict(method_usage),
    }


def write_report(summary: dict):
    lines = [
        "# Retriever Training Set Report",
        "",
        "Purpose:",
        "Create hard-negative triplets and positive pairs for domain-adaptive retriever fine-tuning.",
        "",
        "## Summary",
        "",
        f"- Profile: `{summary['profile']}`",
        f"- Queries: `{summary['queries']}`",
        f"- Pairs: `{summary['pairs']}`",
        f"- Triplets: `{summary['triplets']}`",
        f"- Unique positives: `{summary['unique_positive_refs']}`",
        f"- Unique negatives: `{summary['unique_negative_refs']}`",
        f"- Benchmarks: `{summary['benchmark_counts']}`",
        f"- Languages: `{summary['language_counts']}`",
        f"- Pair splits: `{summary['pair_splits']}`",
        f"- Triplet splits: `{summary['triplet_splits']}`",
        f"- Negative strategies: `{summary['negative_strategies']}`",
        f"- Method usage: `{summary['method_usage']}`",
        "",
        "## Output Files",
        "",
        f"- Corpus: `{CORPUS_PATH}`",
        f"- Pairs: `{PAIRS_PATH}`",
        f"- Triplets: `{TRIPLETS_PATH}`",
        f"- Summary: `{SUMMARY_PATH}`",
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    mappings = load_mappings(PROFILE_KEY)
    ref_to_item = build_ref_index(mappings)
    judgments = load_all_judgments()
    validate_judgments(judgments, mappings)

    context = {
        "profile": PROFILE_KEY,
        "mappings": mappings,
        "semantic_engine": None,
        "bm25_index": None,
        "hybrid_alpha": 0.6,
        "hybrid_candidates": 100,
    }
    warm_context(MINING_METHODS, context)

    corpus_rows = build_corpus_rows(mappings)
    pair_rows = build_pair_rows(judgments, ref_to_item)

    triplet_rows = []
    seen_triplets = set()
    for judgment in judgments:
        pool = collect_candidate_pool(judgment, context)
        for row in choose_negatives(judgment, pool, mappings, ref_to_item):
            key = (row["query"], row["positive_ref"], row["negative_ref"])
            if key in seen_triplets:
                continue
            seen_triplets.add(key)
            triplet_rows.append(row)

    summary = summarize(judgments, pair_rows, triplet_rows)

    write_jsonl(CORPUS_PATH, corpus_rows)
    write_jsonl(PAIRS_PATH, pair_rows)
    write_jsonl(TRIPLETS_PATH, triplet_rows)
    write_json(SUMMARY_PATH, summary)
    write_report(summary)

    print(f"[OK] Wrote retriever corpus to {CORPUS_PATH}")
    print(f"[OK] Wrote retriever pairs to {PAIRS_PATH}")
    print(f"[OK] Wrote retriever triplets to {TRIPLETS_PATH}")
    print(f"[OK] Wrote training summary to {SUMMARY_PATH}")
    print(f"[OK] Wrote report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
