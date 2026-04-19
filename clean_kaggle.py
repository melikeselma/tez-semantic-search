import html
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent

IN_PATH = ROOT / "data" / "raw" / "kaggle" / "raw_kaggle.jsonl"
OUT_PATH = ROOT / "data" / "processed" / "kaggle_clean.jsonl"

# Keep records that have at least some Kaggle metadata beyond title. Rich
# descriptions are flagged separately so the evaluation stage can filter them.
MIN_COMBINED_CHARS = 20

RE_CODE = re.compile(r"```.*?```", re.DOTALL)
RE_HTML_TAG = re.compile(r"<[^>]+>")
RE_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
RE_IMAGE_MD = re.compile(r"!\[([^\]]*)\]\([^)]+\)")
RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_MULTI_WS = re.compile(r"\s+")


def fix_encoding(text: str) -> str:
    replacements = {
        "Ã¢â‚¬â„¢": "'",
        "Ã¢â‚¬Å“": '"',
        "Ã¢â‚¬ï¿½": '"',
        "Ã¢â‚¬â€œ": "-",
        "Ã¢â‚¬â€": "-",
        "Ã¢â‚¬Â¦": "...",
        "Ã‚": "",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text


def clean_text(text: str) -> str:
    text = html.unescape(text or "")
    text = unicodedata.normalize("NFKC", text)
    text = fix_encoding(text)
    text = RE_CODE.sub(" ", text)
    text = RE_IMAGE_MD.sub(" ", text)
    text = RE_MD_LINK.sub(r"\1", text)
    text = RE_HTML_TAG.sub(" ", text)
    text = RE_URL.sub(" <URL> ", text)
    text = RE_MULTI_WS.sub(" ", text)
    return text.strip()


def join_keywords(keywords) -> str:
    if not isinstance(keywords, list):
        return ""
    cleaned = [clean_text(str(item)) for item in keywords if str(item).strip()]
    cleaned = [item for item in cleaned if item]
    return ", ".join(cleaned)


def build_description(rec: dict) -> tuple[str, list[str]]:
    notes = []
    parts = []

    subtitle = clean_text(rec.get("subtitle", ""))
    if subtitle:
        parts.append(subtitle)
        notes.append("subtitle")

    description = clean_text(rec.get("description", ""))
    if description:
        parts.append(description)
        notes.append("description")

    keywords = join_keywords(rec.get("keywords"))
    if keywords:
        parts.append(f"Keywords: {keywords}")
        notes.append("keywords")

    return " ".join(parts).strip(), notes


def quality_flags(description: str, notes: list[str]) -> list[str]:
    flags = []
    if "description" not in notes:
        flags.append("no_long_description")
    if notes == ["keywords"]:
        flags.append("keyword_only")
    if len(description) < 80:
        flags.append("short_description")
    return flags


def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"{IN_PATH} not found. Run fetch_kaggle_descriptions.py first.")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    total = kept = dropped_no_metadata = dropped_short = 0
    flag_counts = Counter()

    with IN_PATH.open("r", encoding="utf-8") as f_in, OUT_PATH.open(
        "w", encoding="utf-8", newline="\n"
    ) as f_out:
        for line in f_in:
            if not line.strip():
                continue
            total += 1
            rec = json.loads(line)

            title = clean_text(rec.get("title", "")) or "Untitled Dataset"
            description, notes = build_description(rec)
            if not description:
                dropped_no_metadata += 1
                continue

            combined_text = f"{title}. {description}".strip()
            if len(combined_text) < MIN_COMBINED_CHARS:
                dropped_short += 1
                continue

            flags = quality_flags(description, notes)
            flag_counts.update(flags)

            out_rec = {
                "source": "kaggle",
                "ref": rec.get("ref"),
                "title": title,
                "description": description,
                "text": combined_text,
                "url": rec.get("url"),
                "keywords": rec.get("keywords") or [],
                "license": rec.get("license") or "",
                "cleaning_notes": notes,
                "quality_flags": flags,
                "description_len_chars": len(description),
                "description_len_words": len(description.split()),
            }

            f_out.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            kept += 1

    print("[OK] Kaggle cleaning complete")
    print(f"  input={total}")
    print(f"  kept={kept}")
    print(f"  dropped_no_metadata={dropped_no_metadata}")
    print(f"  dropped_short={dropped_short}")
    print(f"  quality_flags={dict(flag_counts)}")
    print(f"  output={OUT_PATH}")


if __name__ == "__main__":
    main()
