import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
EVAL_DIR = BASE_DIR / "data" / "evaluation"
OUTPUT_PATH = EVAL_DIR / "relevance_judgments_rq1.json"

SOURCES = [
    {
        "path": EVAL_DIR / "relevance_judgments.json",
        "benchmark": "general_keyword",
        "query_style": "keyword",
        "language": "en",
    },
    {
        "path": EVAL_DIR / "relevance_judgments_en_sentences.json",
        "benchmark": "general_sentence",
        "query_style": "sentence",
        "language": "en",
    },
    {
        "path": EVAL_DIR / "relevance_judgments_earthquake_sentences.json",
        "benchmark": "earthquake_sentence",
        "query_style": "sentence",
        "language": "en",
    },
]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main():
    merged = []

    for source in SOURCES:
        records = load_json(source["path"])
        for record in records:
            enriched = dict(record)
            enriched["benchmark"] = source["benchmark"]
            enriched["query_style"] = source["query_style"]
            enriched["language"] = source["language"]
            merged.append(enriched)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(merged, handle, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote {len(merged)} RQ1 judgments to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
