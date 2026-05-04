"""Quick regression harness for the live semantic search stack.

Runs a curated set of queries (in-corpus, OOD, English, Turkish) across the
available retrieval profiles and methods so we can diff results before/after
each fix instead of eyeballing the UI.

Usage:
    python dev_tools/regression_probe.py             # human-readable report
    python dev_tools/regression_probe.py --json out  # machine-readable diff dump

The harness intentionally stays out of the main package so it can be deleted
once a real evaluation suite is in place.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bm25 import BM25Index
from hybrid import search as hybrid_search
from query_understanding import build_query_plan
from search import load_search_engine, search as semantic_search

# Curated probe set. Mix of:
#   * in-corpus topics (chess, fingerprint, sentiment) -> should rank well
#   * out-of-domain topics (spider, mushroom recipes) -> should weak-match
#   * thesis-curated domains (house price, fraud)     -> regression guard
#   * Turkish queries                                  -> language coverage
PROBES = [
    # OOD English: corpus has zero or near-zero matching docs.
    {"id": "spider_ood", "query": "i want to research on spiders", "language": "en", "expected": "weak"},
    {"id": "spider_species_ood", "query": "i want to research on spider species", "language": "en", "expected": "weak"},
    {"id": "mushroom_in", "query": "mushroom dataset for classification", "language": "en", "expected": "mushroom"},
    {"id": "mushroom_recipe_ood", "query": "vegetarian mushroom recipes", "language": "en", "expected": "weak"},
    {"id": "knitting_ood", "query": "knitting patterns and yarn projects", "language": "en", "expected": "weak"},
    {"id": "wine_ood", "query": "wine tasting notes and grape varietals", "language": "en", "expected": "weak"},
    {"id": "fraud_ood", "query": "credit card fraud detection dataset", "language": "en", "expected": "weak"},
    {"id": "crop_ood", "query": "I want to study how weather affects crop production", "language": "en", "expected": "weak"},
    {"id": "fingerprint_ood", "query": "a dataset of fingerprint images for biometric recognition", "language": "en", "expected": "weak"},
    # In-corpus English: the corpus has real targets, ranking should surface them.
    {"id": "chess", "query": "i want to find a dataset about chess", "language": "en", "expected": "chess"},
    {"id": "house_price", "query": "I want to find data for predicting house prices", "language": "en", "expected": "assessor"},
    {"id": "music", "query": "classical music recordings dataset", "language": "en", "expected": "maestro"},
    {"id": "diabetes", "query": "I want to research diabetes prediction using patient records", "language": "en", "expected": "diabetes"},
    {"id": "urban_change", "query": "I want to study how cities change using images taken from above", "language": "en", "expected": "satellite"},
    {"id": "sentiment_tweets", "query": "i need a dataset for sentiment analysis on tweets", "language": "en", "expected": "sentiment"},
    {"id": "hate_speech", "query": "harmful language and hate speech in social media", "language": "en", "expected": "toxic"},
    {"id": "earthquake", "query": "earthquake seismic activity historical events", "language": "en", "expected": "earthquake"},
    {"id": "deepfake", "query": "deepfake detection audio samples", "language": "en", "expected": "deepfake"},
    {"id": "movie_reviews", "query": "movie reviews with star ratings", "language": "en", "expected": "review"},
    {"id": "satellite_landcover", "query": "satellite imagery land cover classification", "language": "en", "expected": "satellite"},
    # Edge cases: short / very long / question form / multi-aspect.
    {"id": "very_short", "query": "tweets", "language": "en", "expected": "tweet"},
    {"id": "single_token_ood", "query": "spider", "language": "en", "expected": "weak"},
    {"id": "question", "query": "what dataset can I use for fine-tuning a code LLM?", "language": "en", "expected": "code"},
    {"id": "multi_aspect", "query": "satellite images of urban areas with weather metadata", "language": "en", "expected": "satellite"},
    # Turkish.
    {"id": "fraud_tr", "query": "Finansal dolandırıcılık tespiti için veri seti arıyorum", "language": "tr", "expected": "fraud"},
    {"id": "weather_tr", "query": "Yağmurlu hava tahmini için veri seti", "language": "tr", "expected": "weather"},
    {"id": "earthquake_tr", "query": "Deprem sonrası yardım için veri", "language": "tr", "expected": "earthquake"},
    {"id": "spider_tr", "query": "Örümcek türleri hakkında veri seti", "language": "tr", "expected": "weak"},
    {"id": "duygu_tr", "query": "Kullanıcı yorumlarından duygu analizi yapmak istiyorum", "language": "tr", "expected": "sentiment"},
    # --- Stress tier: ambiguity, code-switching, jargon, typos, multi-aspect ---
    {"id": "ambig_apple", "query": "apple dataset", "language": "en", "expected": "weak"},
    {"id": "ambig_python", "query": "python dataset for tutorials", "language": "en", "expected": "weak"},
    {"id": "code_switch", "query": "Türkçe sentiment analysis veri seti", "language": "tr", "expected": "sentiment"},
    {"id": "jargon_xray", "query": "chest x-ray pneumonia detection", "language": "en", "expected": "pneumonia"},
    {"id": "jargon_ecg", "query": "ECG signal arrhythmia classification", "language": "en", "expected": "weak"},
    {"id": "negation", "query": "datasets that are not about images", "language": "en", "expected": "weak"},
    {"id": "typo_mushroom", "query": "musrhoom dataset", "language": "en", "expected": "weak"},
    {"id": "typo_chess", "query": "datset about chess", "language": "en", "expected": "chess"},
    {"id": "single_char", "query": "x", "language": "en", "expected": "weak"},
    {"id": "stopword_only", "query": "i want a dataset", "language": "en", "expected": "weak"},
    {"id": "ml_jargon", "query": "few-shot learning benchmark", "language": "en", "expected": "weak"},
    {"id": "verbose", "query": "I am writing my master thesis on natural language processing applied to political tweets and I need a labelled corpus that contains tweets in Turkish about elections, opinions, and political stance with annotation for sentiment polarity, ideally collected during the past two years", "language": "en", "expected": "tweet"},
    {"id": "rag_methodology", "query": "datasets for evaluating retrieval augmented generation systems", "language": "en", "expected": "weak"},
    {"id": "image_seg", "query": "image segmentation dataset", "language": "en", "expected": "segmentation"},
    {"id": "tr_question", "query": "Hangi veri setleri tıbbi görüntü içerir?", "language": "tr", "expected": "medical"},
    {"id": "diverse_kaggle", "query": "stock market price daily", "language": "en", "expected": "stock"},
    {"id": "multi_lang", "query": "multilingual translation parallel corpus", "language": "en", "expected": "translation"},
]

DEFAULT_PROFILES = ("minilm", "e5_base", "minilm_ft")
DEFAULT_METHODS = ("semantic", "hybrid")


def short_label(item: dict) -> str:
    title = (item.get("title") or "").strip()
    ref = (item.get("ref") or "").strip()
    return f"{ref or '?'} :: {title[:48]}"


def run_probe(probe: dict, engines: dict, bm25_indices: dict, profile: str, method: str, top_k: int = 5, enable_cross_encoder: bool = False):
    model, index, mappings = engines[profile]
    plan = build_query_plan(probe["query"])
    if method == "semantic":
        rows = semantic_search(
            probe["query"], model, index, mappings,
            top_k=top_k, profile_key=profile, query_plan=plan,
            enable_cross_encoder=enable_cross_encoder,
        )
    elif method == "hybrid":
        rows = hybrid_search(
            probe["query"], model, index, mappings, bm25_indices[profile],
            top_k=top_k, profile_key=profile, query_plan=plan,
            enable_cross_encoder=enable_cross_encoder,
        )
    else:
        raise ValueError(method)
    return plan, rows


def format_text_report(report: dict) -> str:
    lines = []
    for probe_id, probe_block in report.items():
        lines.append("=" * 78)
        lines.append(f"[{probe_id}] {probe_block['query']}  (expected~{probe_block['expected']})")
        plan = probe_block["plan"]
        lines.append(
            f"  intent='{plan['intent_body']}'  domains={plan['domains']}  concepts={plan['concept_terms'][:4]}"
        )
        for combo, rows in probe_block["runs"].items():
            lines.append(f"  -- {combo}")
            for score, item in rows:
                lines.append(f"     {score:0.3f}  {short_label(item)}")
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", nargs="*", default=list(DEFAULT_PROFILES))
    parser.add_argument("--methods", nargs="*", default=list(DEFAULT_METHODS))
    parser.add_argument("--cross-encoder", action="store_true", help="Also run with cross-encoder rerank ON")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--json", type=Path, default=None, help="Write the full report as JSON to this path")
    parser.add_argument("--filter", default=None, help="Only run probes whose id contains this substring")
    parser.add_argument("--disable-hardcoded-rules", action="store_true", help="Empty PHRASE_RULES/COMBO_RULES/TOKEN_RULES before probing")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.disable_hardcoded_rules:
        import query_understanding as qu
        qu.PHRASE_RULES.clear()
        qu.COMBO_RULES.clear()
        qu.TOKEN_RULES.clear()

    engines = {profile: load_search_engine(profile) for profile in args.profiles}
    bm25_indices = {profile: BM25Index(engines[profile][2]) for profile in args.profiles}

    report = {}
    for probe in PROBES:
        if args.filter and args.filter not in probe["id"]:
            continue
        block = {
            "query": probe["query"],
            "expected": probe["expected"],
            "language": probe["language"],
            "runs": {},
        }
        plan = None
        for profile in args.profiles:
            for method in args.methods:
                for ce in ([False, True] if args.cross_encoder else [False]):
                    plan, rows = run_probe(probe, engines, bm25_indices, profile, method, args.top_k, ce)
                    combo = f"{profile}|{method}|ce={'on' if ce else 'off'}"
                    block["runs"][combo] = [(float(score), item) for score, item in rows]
        block["plan"] = plan
        report[probe["id"]] = block

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if args.json:
        serialisable = {}
        for probe_id, block in report.items():
            serialisable[probe_id] = {
                "query": block["query"],
                "expected": block["expected"],
                "language": block["language"],
                "plan": {
                    "intent_body": block["plan"]["intent_body"],
                    "focus_terms": block["plan"]["focus_terms"],
                    "concept_terms": block["plan"]["concept_terms"],
                    "domains": block["plan"]["domains"],
                    "semantic_aspects": [a["key"] for a in block["plan"]["semantic_aspects"]],
                },
                "runs": {
                    combo: [
                        {
                            "score": score,
                            "ref": item.get("ref"),
                            "title": item.get("title"),
                            "source": item.get("source"),
                            "weak_match": bool(item.get("weak_match")),
                        }
                        for score, item in rows
                    ]
                    for combo, rows in block["runs"].items()
                },
            }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(serialisable, indent=2, ensure_ascii=False), encoding="utf-8")

    print(format_text_report(report))


if __name__ == "__main__":
    main()
