import json
from pathlib import Path
from typing import Dict, Iterable, Optional

ROOT = Path(r"C:\Users\user\Desktop\Tez")

HF_IN = ROOT / "data" / "processed" / "hf_clean.jsonl"
KG_IN = ROOT / "data" / "processed" / "kaggle_clean.jsonl"

OUT = ROOT / "data" / "processed" / "descriptions_clean.jsonl"
QC = ROOT / "reports" / "qc_merge.txt"


def iter_jsonl(p: Path) -> Iterable[Dict]:
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def pick_text(rec: Dict) -> Optional[str]:
    # Öncelik sırası
    for k in ["text", "description", "readme", "content", "body"]:
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def normalize(rec: Dict, source_hint: str) -> Optional[Dict]:
    text = pick_text(rec)
    if not text:
        return None

    source = rec.get("source") or source_hint

    return {
        "source": source,
        "ref": rec.get("ref") or rec.get("id") or rec.get("name"),
        "title": rec.get("title"),
        "text": text,
        "url": rec.get("url"),
    }


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    QC.parent.mkdir(parents=True, exist_ok=True)

    total_in = 0
    total_out = 0
    dropped = 0
    dropped_sources = {}

    with OUT.open("w", encoding="utf-8", newline="\n") as out:
        for path, hint in [(HF_IN, "huggingface"), (KG_IN, "kaggle")]:
            if not path.exists():
                continue

            for rec in iter_jsonl(path):
                total_in += 1
                norm = normalize(rec, hint)
                if norm is None:
                    dropped += 1
                    s = rec.get("source") or hint
                    dropped_sources[s] = dropped_sources.get(s, 0) + 1
                    continue

                out.write(json.dumps(norm, ensure_ascii=False) + "\n")
                total_out += 1

    with QC.open("w", encoding="utf-8") as f:
        f.write(f"total_in={total_in}\n")
        f.write(f"total_out={total_out}\n")
        f.write(f"dropped_no_text={dropped}\n")
        f.write(f"dropped_by_source={json.dumps(dropped_sources, ensure_ascii=False)}\n")

    print(f"[DONE] wrote: {OUT}")
    print(f"[QC] total_in={total_in} total_out={total_out} dropped_no_text={dropped}")
    print(f"[QC] report: {QC}")


if __name__ == "__main__":
    main()