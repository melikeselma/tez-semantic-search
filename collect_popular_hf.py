"""Fetch metadata for a curated list of popular HuggingFace datasets.

Appends new records to data/raw/huggingface/raw_hf.jsonl in the same format as
collect_hf.py so the rest of the pipeline (clean_hf -> merge_all -> normalize
-> build_faiss_index) can be reused unchanged.
"""
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
RAW_PATH = ROOT / "data" / "raw" / "huggingface" / "raw_hf.jsonl"

# Curated list of widely used / canonical HuggingFace datasets across vision,
# text classification, QA, summarization, translation, ASR, language modelling
# and reasoning benchmarks.
POPULAR_HF_DATASETS = [
    # NLU benchmarks
    "glue", "super_glue", "anli", "snli", "multi_nli", "xnli", "paws", "paws-x",
    # QA
    "squad", "squad_v2", "natural_questions", "trivia_qa", "hotpot_qa",
    "ms_marco", "drop", "boolq", "race", "sciq",
    # Commonsense / reasoning
    "hellaswag", "piqa", "winogrande", "openbookqa", "ai2_arc", "commonsense_qa",
    "social_i_qa", "cosmos_qa",
    # Text classification & sentiment
    "imdb", "ag_news", "dbpedia_14", "yelp_polarity", "yelp_review_full",
    "amazon_polarity", "rotten_tomatoes", "sst2", "emotion", "go_emotions",
    "tweet_eval", "trec", "banking77",
    # Summarization
    "cnn_dailymail", "xsum", "multi_news", "samsum", "dialogsum", "billsum",
    "big_patent", "newsroom",
    # Translation
    "wmt14", "wmt16", "wmt19", "opus100", "ted_talks_iwslt",
    # Language modelling / pretraining
    "wikitext", "bookcorpus", "openwebtext", "the_pile", "c4",
    # NER / sequence labelling
    "conll2003", "wnut_17",
    # Dialogue
    "daily_dialog", "empathetic_dialogues", "persona_chat",
    # Vision
    "mnist", "fashion_mnist", "cifar10", "cifar100", "imagenet-1k", "food101",
    "beans", "cats_vs_dogs",
    # Speech
    "librispeech_asr", "common_voice", "voxpopuli",
    # Math / code
    "gsm8k", "math_dataset", "code_search_net", "humaneval",
    # Multilingual / massive
    "mc4", "oscar",
    # Hate / toxicity
    "hate_speech18", "jigsaw_toxicity_pred",
    # Captioning
    "conceptual_captions",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_existing_ids() -> set:
    existing = set()
    if not RAW_PATH.exists():
        return existing
    with RAW_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                sid = rec.get("source_id")
                if sid:
                    existing.add(sid)
            except json.JSONDecodeError:
                continue
    return existing


def fetch_dataset_info(dataset_id: str) -> dict | None:
    url = f"https://huggingface.co/api/datasets/{dataset_id}"
    try:
        r = requests.get(url, params={"full": "true"}, timeout=15)
    except requests.RequestException as exc:
        print(f"[WARN] {dataset_id}: request failed: {exc}")
        return None
    if r.status_code != 200:
        print(f"[WARN] {dataset_id}: status {r.status_code}")
        return None
    try:
        return r.json()
    except ValueError:
        print(f"[WARN] {dataset_id}: bad JSON")
        return None


def fetch_readme(dataset_id: str) -> str:
    for branch in ("main", "master"):
        url = f"https://huggingface.co/datasets/{dataset_id}/raw/{branch}/README.md"
        try:
            r = requests.get(url, timeout=15)
        except requests.RequestException:
            continue
        if r.status_code == 200 and r.text.strip():
            return r.text
    return ""


def extract_description(info: dict, readme: str) -> str:
    """Use the dataset card README when available; otherwise fall back to
    description fields on the API response."""
    if readme:
        lines = readme.split("\n")
        return "\n".join(lines[:40]).strip()
    desc = info.get("description") or info.get("cardData", {}).get("description") or ""
    return desc.strip()


def build_record(dataset_id: str, info: dict, readme: str) -> dict | None:
    raw_desc = extract_description(info, readme)
    if not raw_desc:
        return None

    tags = info.get("tags") or []
    card = info.get("cardData") or {}
    license_value = card.get("license") or ""
    if isinstance(license_value, list):
        license_value = license_value[0] if license_value else ""

    return {
        "id": str(uuid.uuid4()),
        "source": "huggingface",
        "source_id": dataset_id,
        "title": dataset_id,
        "raw_description": raw_desc,
        "url": f"https://huggingface.co/datasets/{dataset_id}",
        "tags": tags,
        "license": str(license_value or ""),
        "language": "",
        "collected_at": utc_now(),
    }


def main():
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = load_existing_ids()
    print(f"[START] existing records: {len(existing)}")

    added = skipped = failed = 0
    with RAW_PATH.open("a", encoding="utf-8", newline="\n") as out:
        for dataset_id in POPULAR_HF_DATASETS:
            if dataset_id in existing:
                skipped += 1
                continue
            info = fetch_dataset_info(dataset_id)
            if not info:
                failed += 1
                continue
            readme = fetch_readme(dataset_id)
            rec = build_record(dataset_id, info, readme)
            if not rec:
                failed += 1
                print(f"[WARN] {dataset_id}: no description, skipped")
                continue
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            existing.add(dataset_id)
            added += 1
            if added % 10 == 0:
                print(f"[OK] added={added} (last: {dataset_id})")
            time.sleep(0.25)

    print(f"[DONE] added={added}, skipped={skipped}, failed={failed}")
    print(f"[OUT] {RAW_PATH}")


if __name__ == "__main__":
    main()
