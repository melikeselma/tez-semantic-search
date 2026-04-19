import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

ROOT = Path(__file__).resolve().parent

HF_IN = ROOT / "data" / "processed" / "hf_clean.jsonl"
KG_IN = ROOT / "data" / "processed" / "kaggle_clean.jsonl"
OUT = ROOT / "data" / "processed" / "descriptions_clean.jsonl"


TEXT_FIELDS = (
    "clean_description_basic",
    "clean_description",
    "description",
    "text",
    "raw_description",
    "body",
)


def iter_jsonl(path: Path) -> Iterable[Dict]:
    if not path.exists():
        print(f"[WARN] Missing input: {path}")
        return

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[WARN] Skipping invalid JSON in {path.name}:{line_no}: {exc}")


def pick_text(rec: Dict) -> Optional[str]:
    for field in TEXT_FIELDS:
        value = rec.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def pick_ref(rec: Dict, source: str) -> Optional[str]:
    for field in ("source_id", "ref", "id", "name"):
        value = rec.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    title = rec.get("title")
    if isinstance(title, str) and title.strip():
        return f"{source}:{title.strip()}"
    return None


def pick_url(rec: Dict, source: str, ref: Optional[str]) -> Optional[str]:
    url = rec.get("url")
    if isinstance(url, str) and url.strip():
        return url.strip()
    if source == "kaggle" and ref:
        return f"https://www.kaggle.com/datasets/{ref}"
    if source == "huggingface" and ref:
        return f"https://huggingface.co/datasets/{ref}"
    return None


def pick_keywords(rec: Dict) -> list[str]:
    for field in ("keywords", "tags"):
        value = rec.get(field)
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str) and value.strip():
            return [part.strip() for part in value.split(",") if part.strip()]
    return []


def pick_license(rec: Dict) -> str:
    value = rec.get("license")
    if isinstance(value, str) and value.strip():
        return value.strip()
    for tag in pick_keywords(rec):
        if tag.startswith("license:"):
            return tag.split(":", 1)[1].strip()
    return ""


def normalize(rec: Dict, source_hint: str) -> Optional[Dict]:
    source = (rec.get("source") or source_hint or "unknown").strip().lower()
    title = (rec.get("title") or "Untitled Dataset").strip()
    description = pick_text(rec)
    if not description:
        return None

    ref = pick_ref(rec, source)
    url = pick_url(rec, source, ref)

    # The thesis scope uses title + description as the document text.
    text = f"{title}. {description}".strip()

    return {
        "source": source,
        "ref": ref,
        "title": title,
        "description": description,
        "text": text,
        "url": url,
        "keywords": pick_keywords(rec),
        "license": pick_license(rec),
        "quality_flags": rec.get("quality_flags") or [],
        "description_len_chars": rec.get("description_len_chars") or len(description),
        "description_len_words": rec.get("description_len_words") or len(description.split()),
    }


def write_source(path: Path, source_hint: str, out, seen: set) -> Tuple[int, int, int]:
    total = written = skipped = 0

    for rec in iter_jsonl(path) or []:
        total += 1
        norm = normalize(rec, source_hint)
        if not norm:
            skipped += 1
            continue

        key = (norm["source"], norm.get("ref") or norm["title"])
        if key in seen:
            skipped += 1
            continue

        seen.add(key)
        out.write(json.dumps(norm, ensure_ascii=False) + "\n")
        written += 1

    return total, written, skipped


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    totals = {}

    with OUT.open("w", encoding="utf-8", newline="\n") as out:
        for path, source in ((HF_IN, "huggingface"), (KG_IN, "kaggle")):
            total, written, skipped = write_source(path, source, out, seen)
            totals[source] = {"input": total, "written": written, "skipped": skipped}

    total_out = sum(item["written"] for item in totals.values())
    print("[OK] Normalized merge complete")
    for source, counts in totals.items():
        print(
            f"  {source}: input={counts['input']} "
            f"written={counts['written']} skipped={counts['skipped']}"
        )
    print(f"  total_written={total_out}")
    print(f"  output={OUT}")


if __name__ == "__main__":
    main()
