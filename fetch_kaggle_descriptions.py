import json
from pathlib import Path
from typing import Dict, Iterable, Optional

ROOT = Path(__file__).resolve().parent
OUT_PATH = ROOT / "data" / "raw" / "kaggle" / "raw_kaggle.jsonl"

# Metadata downloaded by previous Kaggle API runs. We rebuild raw_kaggle.jsonl
# from these local files so the pipeline is reproducible without new API calls.
METADATA_DIRS = (
    ROOT / "data" / "raw" / "kaggle" / "_tmp_meta",
    ROOT / "temp_kg",
)


def load_metadata(path: Path) -> Optional[Dict]:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[WARN] Skipping unreadable metadata {path}: {exc}")
        return None

    # Some Kaggle API versions write a JSON string inside the JSON file.
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except json.JSONDecodeError:
            return None

    # Some cached files wrap the actual metadata under "info".
    if isinstance(obj, dict) and isinstance(obj.get("info"), dict):
        obj = obj["info"]

    return obj if isinstance(obj, dict) else None


def iter_metadata_files() -> Iterable[Path]:
    for directory in METADATA_DIRS:
        if not directory.exists():
            print(f"[WARN] Missing metadata directory: {directory}")
            continue
        yield from directory.glob("*/dataset-metadata.json")


def as_text(value) -> str:
    return value.strip() if isinstance(value, str) and value.strip() else ""


def extract_ref(meta: Dict, fallback_dir_name: str) -> Optional[str]:
    owner = as_text(meta.get("ownerUser"))
    slug = as_text(meta.get("datasetSlug"))
    if owner and slug:
        return f"{owner}/{slug}"

    fallback = fallback_dir_name.replace("__", "/").strip("/")
    return fallback or None


def extract_license(meta: Dict) -> str:
    licenses = meta.get("licenses")
    if isinstance(licenses, list) and licenses:
        first = licenses[0]
        if isinstance(first, dict):
            return as_text(first.get("name"))
        return as_text(first)
    return ""


def normalize_keywords(value) -> list:
    if not isinstance(value, list):
        return []
    keywords = []
    for item in value:
        if isinstance(item, str) and item.strip():
            keywords.append(item.strip())
    return keywords


def build_record(meta: Dict, metadata_path: Path) -> Optional[Dict]:
    ref = extract_ref(meta, metadata_path.parent.name)
    title = as_text(meta.get("title"))
    if not ref or not title:
        return None

    subtitle = as_text(meta.get("subtitle"))
    description = as_text(meta.get("description"))
    keywords = normalize_keywords(meta.get("keywords"))

    return {
        "source": "kaggle",
        "ref": ref,
        "title": title,
        "subtitle": subtitle,
        "description": description,
        "keywords": keywords,
        "license": extract_license(meta),
        "url": f"https://www.kaggle.com/datasets/{ref}",
        "metadata_path": str(metadata_path.relative_to(ROOT)),
    }


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    seen_refs = set()
    total_files = written = skipped = duplicates = 0

    with OUT_PATH.open("w", encoding="utf-8", newline="\n") as out:
        for metadata_path in iter_metadata_files():
            total_files += 1
            meta = load_metadata(metadata_path)
            if not meta:
                skipped += 1
                continue

            rec = build_record(meta, metadata_path)
            if not rec:
                skipped += 1
                continue

            if rec["ref"] in seen_refs:
                duplicates += 1
                continue

            seen_refs.add(rec["ref"])
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1

    print("[OK] Kaggle raw metadata rebuilt")
    print(f"  metadata_files={total_files}")
    print(f"  written={written}")
    print(f"  duplicates={duplicates}")
    print(f"  skipped={skipped}")
    print(f"  output={OUT_PATH}")


if __name__ == "__main__":
    main()
