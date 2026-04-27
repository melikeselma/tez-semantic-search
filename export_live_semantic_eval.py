import argparse
import csv
import json
import sys
from pathlib import Path

from query_understanding import build_query_plan
from search import load_search_engine, search as semantic_search
from search_profiles import DEFAULT_PROFILE_KEY, get_profile

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_QUERY_SET = BASE_DIR / "data" / "evaluation" / "live_semantic_queries.json"
DEFAULT_OUT_ROOT = BASE_DIR / "reports" / "evaluation" / "live_semantic_queries"


def configure_output():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run live semantic-search evaluation queries and export top-k results for manual review."
    )
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERY_SET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--profile", default=DEFAULT_PROFILE_KEY)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--disable-rerank", action="store_true")
    parser.add_argument("--enable-cross-encoder", action="store_true")
    parser.add_argument("--disable-quality-penalty", action="store_true")
    parser.add_argument("--disable-tr-fusion", action="store_true")
    return parser.parse_args()


def load_query_set(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Query set not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError("Query set must be a JSON list.")
    return rows


def format_list(value):
    if isinstance(value, list):
        return " | ".join(str(item) for item in value)
    return str(value or "")


def build_export_rows(query_rows, profile_key, top_k, enable_rerank, enable_quality_penalty, enable_tr_fusion, enable_cross_encoder):
    model, index, mappings = load_search_engine(profile_key)
    export_rows = []

    for query_row in query_rows:
        query_text = str(query_row.get("query_text") or "").strip()
        if not query_text:
            continue

        query_plan = build_query_plan(query_text)
        results = semantic_search(
            query_text,
            model,
            index,
            mappings,
            top_k=top_k,
            profile_key=profile_key,
            query_plan=query_plan,
            enable_rerank=enable_rerank,
            enable_quality_penalty=enable_quality_penalty,
            enable_tr_fusion=enable_tr_fusion,
            enable_cross_encoder=enable_cross_encoder,
        )

        for rank, (score, item) in enumerate(results, start=1):
            export_rows.append(
                {
                    "query_id": query_row.get("query_id"),
                    "query_text": query_text,
                    "language": query_row.get("language") or query_plan.get("detected_language") or "",
                    "profile": profile_key,
                    "rank": rank,
                    "score": float(score),
                    "title": item.get("title") or "",
                    "source": item.get("source") or "",
                    "url": item.get("url") or "",
                    "quality_flags": item.get("quality_flags") or [],
                    "inferred_domains": item.get("inferred_domains") or [],
                    "inferred_use_cases": item.get("inferred_use_cases") or [],
                    "inferred_modalities": item.get("inferred_modalities") or [],
                    "semantic_quality_note": item.get("semantic_quality_note") or "",
                    "semantic_variant_hits": item.get("semantic_variant_hits"),
                    "semantic_rerank_score": item.get("semantic_rerank_score"),
                    "cross_encoder_score": item.get("cross_encoder_score"),
                    "cross_encoder_score_norm": item.get("cross_encoder_score_norm"),
                    "cross_encoder_model": item.get("cross_encoder_model") or "",
                    "expected_domain": query_row.get("expected_domain") or [],
                    "expected_task": query_row.get("expected_task") or [],
                    "expected_modality": query_row.get("expected_modality") or [],
                    "positive_clues": query_row.get("positive_clues") or [],
                    "known_failure_patterns": query_row.get("known_failure_patterns") or [],
                    "manual_relevance": "",
                    "manual_notes": "",
                }
            )

        if len(results) < top_k:
            for rank in range(len(results) + 1, top_k + 1):
                export_rows.append(
                    {
                        "query_id": query_row.get("query_id"),
                        "query_text": query_text,
                        "language": query_row.get("language") or query_plan.get("detected_language") or "",
                        "profile": profile_key,
                        "rank": rank,
                        "score": "",
                        "title": "",
                        "source": "",
                        "url": "",
                        "quality_flags": [],
                        "inferred_domains": [],
                        "inferred_use_cases": [],
                        "inferred_modalities": [],
                        "semantic_quality_note": "",
                        "semantic_variant_hits": "",
                        "semantic_rerank_score": "",
                        "cross_encoder_score": "",
                        "cross_encoder_score_norm": "",
                        "cross_encoder_model": "",
                        "expected_domain": query_row.get("expected_domain") or [],
                        "expected_task": query_row.get("expected_task") or [],
                        "expected_modality": query_row.get("expected_modality") or [],
                        "positive_clues": query_row.get("positive_clues") or [],
                        "known_failure_patterns": query_row.get("known_failure_patterns") or [],
                        "manual_relevance": "",
                        "manual_notes": "",
                    }
                )

    return export_rows


def write_csv(path: Path, rows):
    if not rows:
        return
    fieldnames = [
        "query_id",
        "query_text",
        "language",
        "profile",
        "rank",
        "score",
        "title",
        "source",
        "url",
        "quality_flags",
        "inferred_domains",
        "inferred_use_cases",
        "inferred_modalities",
        "semantic_quality_note",
        "semantic_variant_hits",
        "semantic_rerank_score",
        "cross_encoder_score",
        "cross_encoder_score_norm",
        "cross_encoder_model",
        "expected_domain",
        "expected_task",
        "expected_modality",
        "positive_clues",
        "known_failure_patterns",
        "manual_relevance",
        "manual_notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "quality_flags": format_list(row.get("quality_flags")),
                    "inferred_domains": format_list(row.get("inferred_domains")),
                    "inferred_use_cases": format_list(row.get("inferred_use_cases")),
                    "inferred_modalities": format_list(row.get("inferred_modalities")),
                    "expected_domain": format_list(row.get("expected_domain")),
                    "expected_task": format_list(row.get("expected_task")),
                    "expected_modality": format_list(row.get("expected_modality")),
                    "positive_clues": format_list(row.get("positive_clues")),
                    "known_failure_patterns": format_list(row.get("known_failure_patterns")),
                }
            )


def main():
    configure_output()
    args = parse_args()
    profile = get_profile(args.profile)
    query_rows = load_query_set(args.queries)
    out_dir = args.out_dir / profile.key
    out_dir.mkdir(parents=True, exist_ok=True)

    export_rows = build_export_rows(
        query_rows,
        profile_key=profile.key,
        top_k=max(1, args.top_k),
        enable_rerank=not args.disable_rerank,
        enable_quality_penalty=not args.disable_quality_penalty,
        enable_tr_fusion=not args.disable_tr_fusion,
        enable_cross_encoder=args.enable_cross_encoder,
    )

    metadata = {
        "profile": profile.key,
        "profile_label": profile.label,
        "model_name": profile.model_name,
        "top_k": max(1, args.top_k),
        "query_count": len(query_rows),
        "enable_rerank": not args.disable_rerank,
        "enable_cross_encoder": args.enable_cross_encoder,
        "enable_quality_penalty": not args.disable_quality_penalty,
        "enable_tr_fusion": not args.disable_tr_fusion,
        "queries_path": str(args.queries),
        "results": export_rows,
    }

    out_json = out_dir / "top10_results.json"
    out_csv = out_dir / "top10_results.csv"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    write_csv(out_csv, export_rows)

    print(f"[OK] Exported live semantic evaluation results for profile={profile.key}")
    print(f"  queries={args.queries}")
    print(f"  rows={len(export_rows)}")
    print(f"  json={out_json}")
    print(f"  csv={out_csv}")


if __name__ == "__main__":
    main()
