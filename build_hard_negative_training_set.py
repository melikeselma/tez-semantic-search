import csv
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

from runtime_env import ensure_model_cache_dirs
from search import load_mappings, load_search_engine, search as semantic_search
from search_profiles import DEFAULT_PROFILE_KEY

BASE_DIR = Path(__file__).resolve().parent
TRAIN_DIR = BASE_DIR / "data" / "training"
REPORT_DIR = BASE_DIR / "reports" / "training"

QUERY_SPEC_PATH = TRAIN_DIR / "hard_negative_query_specs.json"
TRIPLETS_JSONL_PATH = TRAIN_DIR / "retriever_hard_negative_triplets.jsonl"
TRIPLETS_CSV_PATH = TRAIN_DIR / "retriever_hard_negative_triplets.csv"
SUMMARY_PATH = TRAIN_DIR / "retriever_hard_negative_summary.json"
REPORT_PATH = REPORT_DIR / "retriever_hard_negative_report.md"

PROFILE_KEY = DEFAULT_PROFILE_KEY
TOP_SEMANTIC_CANDIDATES = 40
POSITIVES_PER_QUERY = 2
NEGATIVES_PER_POSITIVE = 3
TRAIN_TEXT_MAX_CHARS = 1600
LOW_INFORMATION_FLAGS = {
    "empty",
    "too_short",
    "short_description",
    "keyword_only",
    "metadata_heavy",
    "low_information",
    "term_sparse",
    "no_long_description",
}


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "query_id",
        "query",
        "language",
        "positive_ref",
        "positive_title",
        "positive_selection_quality",
        "negative_ref",
        "negative_title",
        "negative_reason",
        "negative_rank_hint",
        "positive_text",
        "negative_text",
        "positive_clue_hits",
        "negative_clue_hits",
        "hard_negative_clue_hits",
        "positive_quality_flags",
        "negative_quality_flags",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def normalize_text(text: str) -> str:
    text = str(text or "").lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^\w\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def unique_preserve_order(values):
    seen = set()
    ordered = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def build_matching_text(item: dict) -> str:
    parts = [
        item.get("title") or "",
        item.get("semantic_text") or "",
        item.get("description") or "",
        item.get("text") or "",
        item.get("semantic_summary") or "",
        " ".join(item.get("keywords") or []),
        " ".join(item.get("inferred_domains") or []),
        " ".join(item.get("inferred_use_cases") or []),
        " ".join(item.get("inferred_modalities") or []),
        item.get("semantic_quality_note") or "",
    ]
    return normalize_text(" ".join(str(part) for part in parts if part))


def build_training_text(item: dict) -> str:
    text = (
        item.get("semantic_text")
        or item.get("description")
        or item.get("text")
        or item.get("semantic_summary")
        or item.get("title")
        or ""
    )
    return str(text).strip()[:TRAIN_TEXT_MAX_CHARS]


def phrase_hits(text: str, clues) -> list[str]:
    hits = []
    for clue in clues or []:
        normalized = normalize_text(clue)
        if normalized and normalized in text:
            hits.append(str(clue))
    return unique_preserve_order(hits)


def normalized_value_set(values) -> set[str]:
    return {normalize_text(value) for value in values or [] if normalize_text(value)}


def overlap_hits(values, expected_values) -> int:
    return len(normalized_value_set(values) & normalized_value_set(expected_values))


def analyze_candidate(item: dict, spec: dict) -> dict:
    matching_text = build_matching_text(item)
    core_positive_hits = phrase_hits(matching_text, spec.get("core_positive_clues") or [])
    positive_hits = phrase_hits(matching_text, spec.get("positive_clues") or [])
    negative_hits = phrase_hits(matching_text, spec.get("hard_negative_clues") or [])
    domain_hits = overlap_hits(item.get("inferred_domains") or [], spec.get("expected_domains") or [])
    use_case_hits = overlap_hits(item.get("inferred_use_cases") or [], spec.get("expected_use_cases") or [])
    modality_hits = overlap_hits(item.get("inferred_modalities") or [], spec.get("expected_modalities") or [])
    quality_flags = [str(flag) for flag in (item.get("quality_flags") or []) if str(flag)]
    low_information_hits = [flag for flag in quality_flags if normalize_text(flag) in LOW_INFORMATION_FLAGS]
    positive_score = (
        4.0 * len(core_positive_hits)
        + 1.2 * max(len(positive_hits) - len(core_positive_hits), 0)
        + 1.4 * domain_hits
        + 1.0 * use_case_hits
        + 0.8 * modality_hits
        - 2.6 * len(negative_hits)
        - 1.0 * len(low_information_hits)
    )
    return {
        "item": item,
        "matching_text": matching_text,
        "training_text": build_training_text(item),
        "core_positive_hits": core_positive_hits,
        "positive_hits": positive_hits,
        "negative_hits": negative_hits,
        "domain_hits": domain_hits,
        "use_case_hits": use_case_hits,
        "modality_hits": modality_hits,
        "quality_flags": quality_flags,
        "low_information_flags": low_information_hits,
        "positive_score": round(positive_score, 6),
    }


def is_strong_positive(metrics: dict) -> bool:
    if metrics["negative_hits"]:
        return False
    if metrics["positive_score"] >= 6.0:
        return True
    return (
        len(metrics["core_positive_hits"]) >= 2
        or (
            len(metrics["core_positive_hits"]) >= 1
            and metrics["modality_hits"] >= 1
            and (metrics["domain_hits"] >= 1 or metrics["use_case_hits"] >= 1)
        )
    )


def selection_quality(metrics: dict) -> str:
    if is_strong_positive(metrics):
        return "strong"
    if not metrics["negative_hits"] and (
        metrics["positive_score"] >= 2.5 and (
            metrics["core_positive_hits"]
            or len(metrics["positive_hits"]) >= 2
        )
    ):
        return "weak"
    return "fallback"


def choose_positive_examples(spec: dict, mappings: dict) -> list[dict]:
    primary_candidates = []
    fallback_candidates = []
    for item in mappings.values():
        metrics = analyze_candidate(item, spec)
        if not metrics["training_text"]:
            continue
        if metrics["negative_hits"]:
            continue
        if metrics["core_positive_hits"] or len(metrics["positive_hits"]) >= 2:
            primary_candidates.append(metrics)
        elif metrics["positive_score"] > 0:
            fallback_candidates.append(metrics)

    primary_candidates.sort(
        key=lambda row: (
            is_strong_positive(row),
            row["positive_score"],
            len(row["core_positive_hits"]),
            len(row["positive_hits"]),
            row["domain_hits"] + row["use_case_hits"] + row["modality_hits"],
        ),
        reverse=True,
    )
    fallback_candidates.sort(
        key=lambda row: (
            row["positive_score"],
            len(row["positive_hits"]),
            row["domain_hits"] + row["use_case_hits"] + row["modality_hits"],
        ),
        reverse=True,
    )

    selected = []
    seen_refs = set()
    for metrics in primary_candidates + fallback_candidates:
        ref = str(metrics["item"].get("ref") or "")
        if not ref or ref in seen_refs:
            continue
        seen_refs.add(ref)
        if selection_quality(metrics) == "fallback" and selected:
            continue
        selected.append(metrics)
        if len(selected) >= POSITIVES_PER_QUERY:
            break
    return selected


def semantic_rank_score(rank: int) -> float:
    if rank <= 0:
        return 0.0
    return max(0.0, 1.0 - ((rank - 1) / max(TOP_SEMANTIC_CANDIDATES, 1)))


def choose_negative_examples(spec: dict, positives: list[dict], semantic_results) -> list[dict]:
    positive_refs = {row["item"].get("ref") for row in positives if row["item"].get("ref")}
    candidates = []

    for rank, (score, item) in enumerate(semantic_results, start=1):
        ref = item.get("ref")
        if not ref or ref in positive_refs:
            continue
        metrics = analyze_candidate(item, spec)
        if not metrics["training_text"]:
            continue

        wrong_aspects = 0
        if spec.get("expected_domains") and metrics["domain_hits"] == 0:
            wrong_aspects += 1
        if spec.get("expected_use_cases") and metrics["use_case_hits"] == 0:
            wrong_aspects += 1
        if spec.get("expected_modalities") and metrics["modality_hits"] == 0:
            wrong_aspects += 1

        partial_match = (
            bool(metrics["positive_hits"])
            or metrics["domain_hits"] > 0
            or metrics["use_case_hits"] > 0
            or metrics["modality_hits"] > 0
        )
        if not partial_match and not metrics["negative_hits"]:
            continue
        if is_strong_positive(metrics):
            continue

        hardness = (
            1.7 * semantic_rank_score(rank)
            + 1.1 * float(score)
            + 1.3 * len(metrics["negative_hits"])
            + 0.7 * len(metrics["positive_hits"])
            + 0.8 * wrong_aspects
        )
        reason_parts = []
        if metrics["negative_hits"]:
            reason_parts.append("negative_clue_match")
        if partial_match:
            reason_parts.append("partial_semantic_overlap")
        if wrong_aspects:
            reason_parts.append("missing_query_aspects")

        candidates.append(
            {
                "item": item,
                "metrics": metrics,
                "rank_hint": rank,
                "retrieval_score": float(score),
                "hardness": round(hardness, 6),
                "reason": ",".join(reason_parts) or "semantic_neighbor",
            }
        )

    candidates.sort(
        key=lambda row: (
            row["hardness"],
            len(row["metrics"]["negative_hits"]),
            len(row["metrics"]["positive_hits"]),
        ),
        reverse=True,
    )

    selected = []
    seen_refs = set()
    for row in candidates:
        ref = str(row["item"].get("ref") or "")
        if not ref or ref in seen_refs:
            continue
        seen_refs.add(ref)
        selected.append(row)
        if len(selected) >= NEGATIVES_PER_POSITIVE:
            break
    return selected


def build_triples(specs: list[dict], mappings: dict) -> list[dict]:
    ensure_model_cache_dirs()
    model, index, search_mappings = load_search_engine(PROFILE_KEY)
    triples = []

    # Mine hard negatives from semantic top-N neighbors so the training set
    # teaches the bi-encoder to separate semantically close but wrong datasets.
    for spec in specs:
        semantic_results = semantic_search(
            spec["query_text"],
            model,
            index,
            search_mappings,
            top_k=TOP_SEMANTIC_CANDIDATES,
            profile_key=PROFILE_KEY,
            enable_rerank=True,
            enable_tr_fusion=False,
        )
        positives = choose_positive_examples(spec, mappings)
        negatives = choose_negative_examples(spec, positives, semantic_results)

        for positive in positives:
            for negative in negatives:
                triples.append(
                    {
                        "query_id": spec["query_id"],
                        "query": spec["query_text"],
                        "language": spec.get("language") or "",
                        "positive_ref": positive["item"].get("ref") or "",
                        "positive_title": positive["item"].get("title") or "",
                        "positive_selection_quality": selection_quality(positive),
                        "negative_ref": negative["item"].get("ref") or "",
                        "negative_title": negative["item"].get("title") or "",
                        "negative_reason": negative["reason"],
                        "negative_rank_hint": negative["rank_hint"],
                        "positive_text": positive["training_text"],
                        "negative_text": negative["metrics"]["training_text"],
                        "positive_clue_hits": " | ".join(positive["positive_hits"]),
                        "negative_clue_hits": " | ".join(negative["metrics"]["positive_hits"]),
                        "hard_negative_clue_hits": " | ".join(negative["metrics"]["negative_hits"]),
                        "positive_quality_flags": " | ".join(positive["quality_flags"]),
                        "negative_quality_flags": " | ".join(negative["metrics"]["quality_flags"]),
                    }
                )
    return triples


def summarize(triples: list[dict], specs: list[dict]) -> dict:
    per_query = Counter(row["query_id"] for row in triples)
    positive_quality = Counter(row["positive_selection_quality"] for row in triples)
    negative_reasons = Counter()
    unique_positive_refs = set()
    unique_negative_refs = set()
    for row in triples:
        unique_positive_refs.add(row["positive_ref"])
        unique_negative_refs.add(row["negative_ref"])
        for reason in str(row.get("negative_reason") or "").split(","):
            if reason:
                negative_reasons[reason] += 1

    return {
        "profile": PROFILE_KEY,
        "query_specs": str(QUERY_SPEC_PATH),
        "queries": len(specs),
        "triples": len(triples),
        "unique_positive_refs": len(unique_positive_refs),
        "unique_negative_refs": len(unique_negative_refs),
        "triples_per_query": dict(per_query),
        "positive_selection_quality": dict(positive_quality),
        "negative_reasons": dict(negative_reasons),
        "outputs": {
            "jsonl": str(TRIPLETS_JSONL_PATH),
            "csv": str(TRIPLETS_CSV_PATH),
            "summary": str(SUMMARY_PATH),
            "report": str(REPORT_PATH),
        },
    }


def write_report(summary: dict, specs: list[dict]):
    lines = [
        "# Hard-Negative Retriever Training Set",
        "",
        "Purpose:",
        "Create a small sentence-transformers training set from live semantic failure cases.",
        "",
        "Why this matters:",
        "These triples keep the project semantic-search focused while teaching the retriever to push down semantically nearby but wrong datasets.",
        "",
        "## Summary",
        "",
        f"- Profile: `{summary['profile']}`",
        f"- Query specs: `{summary['query_specs']}`",
        f"- Queries: `{summary['queries']}`",
        f"- Triples: `{summary['triples']}`",
        f"- Unique positives: `{summary['unique_positive_refs']}`",
        f"- Unique negatives: `{summary['unique_negative_refs']}`",
        f"- Positive selection quality: `{summary['positive_selection_quality']}`",
        f"- Negative reasons: `{summary['negative_reasons']}`",
        "",
        "## Queries",
        "",
    ]
    for spec in specs:
        lines.append(f"- `{spec['query_id']}`: {spec['query_text']}")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    mappings = load_mappings(PROFILE_KEY)
    with QUERY_SPEC_PATH.open("r", encoding="utf-8") as handle:
        specs = json.load(handle)

    triples = build_triples(specs, mappings)
    summary = summarize(triples, specs)

    write_jsonl(TRIPLETS_JSONL_PATH, triples)
    write_csv(TRIPLETS_CSV_PATH, triples)
    write_json(SUMMARY_PATH, summary)
    write_report(summary, specs)

    print(f"[OK] Wrote hard-negative triplets to {TRIPLETS_JSONL_PATH}")
    print(f"[OK] Wrote training CSV to {TRIPLETS_CSV_PATH}")
    print(f"[OK] Wrote summary to {SUMMARY_PATH}")
    print(f"[OK] Wrote report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
