import json
from pathlib import Path

ROOT = Path(r"C:\Users\user\Desktop\Tez")

HF_PATH = ROOT / "data" / "processed" / "hf_clean.jsonl"
KAGGLE_PATH = ROOT / "data" / "processed" / "kaggle_clean.jsonl"
OUT_PATH = ROOT / "data" / "processed" / "descriptions_clean.jsonl"


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    total = 0

    with open(OUT_PATH, "w", encoding="utf-8") as out:

        for path in [HF_PATH, KAGGLE_PATH]:
            if not path.exists():
                continue

            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    out.write(line)
                    total += 1

    print(f"[DONE] merged total={total}")
    print(f"[OUT] {OUT_PATH}")


if __name__ == "__main__":
    main()