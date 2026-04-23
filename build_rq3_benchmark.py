import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
EVAL_DIR = BASE_DIR / "data" / "evaluation"
OUTPUT_PATH = EVAL_DIR / "relevance_judgments_rq3.json"

SOURCES = [
    {
        "path": EVAL_DIR / "relevance_judgments_rq1.json",
        "study_slice": "english_main",
    },
    {
        "path": EVAL_DIR / "relevance_judgments_rq2.json",
        "study_slice": "cross_source",
    },
    {
        "path": EVAL_DIR / "relevance_judgments_tr.json",
        "study_slice": "tr_subset",
        "benchmark": "turkish_subset",
        "query_style": "keyword",
        "language": "tr",
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
            enriched["study_slice"] = source["study_slice"]
            if "benchmark" in source:
                enriched.setdefault("benchmark", source["benchmark"])
            if "query_style" in source:
                enriched.setdefault("query_style", source["query_style"])
            if "language" in source:
                enriched.setdefault("language", source["language"])
            merged.append(enriched)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(merged, handle, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote {len(merged)} RQ3 judgments to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
