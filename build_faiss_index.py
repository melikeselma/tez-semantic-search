import json
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

ROOT = Path(r"C:\Users\user\Desktop\Tez")

IN_PATH = ROOT / "data" / "processed" / "descriptions_clean.jsonl"
EMB_DIR = ROOT / "data" / "embeddings"
IDX_DIR = ROOT / "data" / "index"

EMB_DIR.mkdir(parents=True, exist_ok=True)
IDX_DIR.mkdir(parents=True, exist_ok=True)

OUT_EMB = EMB_DIR / "descriptions.npy"
OUT_META = EMB_DIR / "meta.jsonl"
OUT_INDEX = IDX_DIR / "faiss.index"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 32


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    rows = list(iter_jsonl(IN_PATH))

    texts = []
    meta = []

    for r in rows:
        t = r.get("text")
        if isinstance(t, str) and t.strip():
            texts.append(t.strip())
            meta.append({
                "source": r.get("source"),
                "ref": r.get("ref"),
                "title": r.get("title"),
                "url": r.get("url"),
            })

    if not texts:
        raise RuntimeError("No valid 'text' found in descriptions_clean.jsonl")

    print(f"[LOAD] {len(texts)} texts ready for embedding")

    model = SentenceTransformer(MODEL_NAME)
    emb = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype("float32")

    np.save(OUT_EMB, emb)

    with OUT_META.open("w", encoding="utf-8", newline="\n") as out:
        for m in meta:
            out.write(json.dumps(m, ensure_ascii=False) + "\n")

    d = emb.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(emb)

    faiss.write_index(index, str(OUT_INDEX))

    print(f"[SAVE] embeddings: {OUT_EMB} shape={emb.shape}")
    print(f"[SAVE] meta: {OUT_META} lines={len(meta)}")
    print(f"[SAVE] index: {OUT_INDEX} ntotal={index.ntotal}")


if __name__ == "__main__":
    main()