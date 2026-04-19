import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent

IN_PATH = ROOT / "data" / "processed" / "descriptions_clean.jsonl"
IDX_DIR = ROOT / "data" / "index"
EMB_DIR = ROOT / "data" / "embeddings"

OUT_INDEX = IDX_DIR / "faiss.index"
OUT_MAPPING = IDX_DIR / "mappings.json"
OUT_INDEX_META = IDX_DIR / "index_metadata.json"
OUT_LEGACY_META = EMB_DIR / "meta.jsonl"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 32


def iter_jsonl(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run normalize_merge.py first.")

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc


def main():
    IDX_DIR.mkdir(parents=True, exist_ok=True)
    EMB_DIR.mkdir(parents=True, exist_ok=True)

    rows = list(iter_jsonl(IN_PATH))
    texts = []
    mappings = {}
    legacy_meta_rows = []
    source_counts = Counter()
    skipped = 0

    for row in rows:
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            skipped += 1
            continue

        index_id = len(texts)
        source = row.get("source") or "unknown"
        source_counts[source] += 1
        clean_text = text.strip()

        item = {
            "source": source,
            "ref": row.get("ref"),
            "title": row.get("title") or "Untitled Dataset",
            "url": row.get("url"),
            "description": row.get("description"),
            "text": clean_text,
            "keywords": row.get("keywords") or [],
            "license": row.get("license") or "",
            "quality_flags": row.get("quality_flags") or [],
            "description_len_chars": row.get("description_len_chars"),
            "description_len_words": row.get("description_len_words"),
        }
        mappings[str(index_id)] = item
        legacy_meta_rows.append({k: item.get(k) for k in ("source", "ref", "title", "url")})
        texts.append(clean_text)

    if not texts:
        raise RuntimeError("No valid text found to index.")

    print(f"[LOAD] source_rows={len(rows)} valid_texts={len(texts)} skipped={skipped}")
    print(f"[LOAD] source_counts={dict(source_counts)}")

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    if index.ntotal != len(mappings):
        raise RuntimeError(
            f"Index/mapping mismatch before write: index={index.ntotal}, "
            f"mappings={len(mappings)}"
        )

    faiss.write_index(index, str(OUT_INDEX))

    with OUT_MAPPING.open("w", encoding="utf-8") as f:
        json.dump(mappings, f, ensure_ascii=False, indent=2)

    with OUT_LEGACY_META.open("w", encoding="utf-8", newline="\n") as f:
        for row in legacy_meta_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    index_metadata = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": MODEL_NAME,
        "input_path": str(IN_PATH),
        "input_rows": len(rows),
        "indexed_rows": index.ntotal,
        "embedding_dimension": dimension,
        "source_counts": dict(source_counts),
        "outputs": {
            "index": str(OUT_INDEX),
            "mappings": str(OUT_MAPPING),
            "legacy_meta": str(OUT_LEGACY_META),
        },
    }
    with OUT_INDEX_META.open("w", encoding="utf-8") as f:
        json.dump(index_metadata, f, ensure_ascii=False, indent=2)

    print("[OK] FAISS index built")
    print(f"  index={OUT_INDEX}")
    print(f"  mappings={OUT_MAPPING}")
    print(f"  legacy_meta={OUT_LEGACY_META}")
    print(f"  indexed_rows={index.ntotal}")


if __name__ == "__main__":
    main()
