import json
import re
import html
from pathlib import Path

ROOT = Path(r"C:\Users\user\Desktop\Tez")

IN_PATH = ROOT / "data" / "raw" / "kaggle" / "raw_kaggle.jsonl"
OUT_PATH = ROOT / "data" / "processed" / "kaggle_clean.jsonl"

MIN_CHARS = 80

RE_CODE = re.compile(r"```.*?```", re.DOTALL)
RE_HTML_TAG = re.compile(r"<[^>]+>")
RE_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
RE_MULTI_WS = re.compile(r"\s+")


def fix_encoding(s: str) -> str:
    replacements = {
        "â€™": "’",
        "â€œ": "“",
        "â€�": "”",
        "â€“": "–",
        "â€”": "—",
        "â€¦": "…",
        "Â": "",
    }
    for bad, good in replacements.items():
        s = s.replace(bad, good)
    return s


def clean_text(s: str) -> str:
    s = html.unescape(s)
    s = fix_encoding(s)
    s = RE_CODE.sub(" ", s)
    s = RE_HTML_TAG.sub(" ", s)
    s = RE_MD_LINK.sub(r"\1", s)
    s = RE_MULTI_WS.sub(" ", s)
    return s.strip()


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    kept = 0

    with open(IN_PATH, "r", encoding="utf-8") as f_in, \
         open(OUT_PATH, "w", encoding="utf-8") as f_out:

        for line in f_in:
            total += 1
            rec = json.loads(line)

            text = clean_text(rec.get("description", ""))

            if len(text) < MIN_CHARS:
                continue

            out_rec = {
                "source": "kaggle",
                "ref": rec.get("ref"),
                "title": rec.get("title"),
                "text": text,
                "url": rec.get("url")
            }

            f_out.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            kept += 1

    print(f"[DONE] total={total}, kept={kept}")
    print(f"[OUT] {OUT_PATH}")


if __name__ == "__main__":
    main()