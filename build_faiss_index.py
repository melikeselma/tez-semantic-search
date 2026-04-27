import json
from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

from runtime_env import ensure_model_cache_dirs
from search_profiles import (
    DEFAULT_PROFILE_KEY,
    get_profile,
    get_profile_paths,
    prepare_document_text,
)

ROOT = Path(__file__).resolve().parent

IN_PATH = ROOT / "data" / "processed" / "descriptions_clean.jsonl"
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


def parse_args():
    parser = ArgumentParser(description="Build a FAISS index for a configured search profile.")
    parser.add_argument("--profile", default=DEFAULT_PROFILE_KEY)
    return parser.parse_args()


def main():
    args = parse_args()
    profile = get_profile(args.profile)
    paths = get_profile_paths(profile.key)
    index_dir = paths["index_dir"]
    embedding_dir = paths["embedding_dir"]
    out_index = paths["index"]
    out_mapping = paths["mappings"]
    out_index_meta = paths["index_metadata"]
    out_legacy_meta = paths["legacy_meta"]

    ensure_model_cache_dirs()

    index_dir.mkdir(parents=True, exist_ok=True)
    embedding_dir.mkdir(parents=True, exist_ok=True)

    rows = list(iter_jsonl(IN_PATH))
    texts = []
    mappings = {}
    legacy_meta_rows = []
    source_counts = Counter()
    skipped = 0

    for row in rows:
        text = row.get("semantic_text") or row.get("text")
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
            "text": row.get("text"),
            "semantic_text": clean_text,
            "semantic_summary": row.get("semantic_summary") or "",
            "semantic_quality_note": row.get("semantic_quality_note") or "",
            "keywords": row.get("keywords") or [],
            "language_hint": row.get("language_hint") or "",
            "metadata_terms": row.get("metadata_terms") or [],
            "inferred_domains": row.get("inferred_domains") or [],
            "inferred_use_cases": row.get("inferred_use_cases") or [],
            "inferred_modalities": row.get("inferred_modalities") or [],
            "license": row.get("license") or "",
            "quality_flags": row.get("quality_flags") or [],
            "description_len_chars": row.get("description_len_chars"),
            "description_len_words": row.get("description_len_words"),
        }
        mappings[str(index_id)] = item
        legacy_meta_rows.append({k: item.get(k) for k in ("source", "ref", "title", "url")})
        texts.append(prepare_document_text(profile.key, clean_text))

    if not texts:
        raise RuntimeError("No valid text found to index.")

    print(f"[LOAD] profile={profile.key}")
    print(f"[LOAD] source_rows={len(rows)} valid_texts={len(texts)} skipped={skipped}")
    print(f"[LOAD] source_counts={dict(source_counts)}")
    print(f"[MODEL] {profile.model_name}")

    model = SentenceTransformer(profile.model_name)
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

    faiss.write_index(index, str(out_index))

    with out_mapping.open("w", encoding="utf-8") as f:
        json.dump(mappings, f, ensure_ascii=False, indent=2)

    with out_legacy_meta.open("w", encoding="utf-8", newline="\n") as f:
        for row in legacy_meta_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    index_metadata = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile": profile.key,
        "profile_label": profile.label,
        "model_name": profile.model_name,
        "input_path": str(IN_PATH),
        "input_rows": len(rows),
        "indexed_rows": index.ntotal,
        "embedding_dimension": dimension,
        "query_prefix": profile.query_prefix,
        "document_prefix": profile.document_prefix,
        "source_counts": dict(source_counts),
        "outputs": {
            "index": str(out_index),
            "mappings": str(out_mapping),
            "legacy_meta": str(out_legacy_meta),
        },
    }
    with out_index_meta.open("w", encoding="utf-8") as f:
        json.dump(index_metadata, f, ensure_ascii=False, indent=2)

    print("[OK] FAISS index built")
    print(f"  profile={profile.key}")
    print(f"  index={out_index}")
    print(f"  mappings={out_mapping}")
    print(f"  legacy_meta={out_legacy_meta}")
    print(f"  indexed_rows={index.ntotal}")


if __name__ == "__main__":
    main()
