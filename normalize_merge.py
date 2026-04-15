import json
from pathlib import Path
from typing import Dict, Iterable, Optional

ROOT = Path(__file__).resolve().parent

# Hem ham hem temizlenmiş yolları kontrol edelim
HF_IN = ROOT / "data" / "processed" / "hf_clean.jsonl"
KG_IN = ROOT / "data" / "raw" / "kaggle" / "raw_kaggle.jsonl" # Kaggle'ı doğrudan hamdan çekiyoruz

OUT = ROOT / "data" / "processed" / "descriptions_clean.jsonl"

def iter_jsonl(p: Path) -> Iterable[Dict]:
    if not p.exists(): return
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line: yield json.loads(line)

def pick_text(rec: Dict) -> Optional[str]:
    # Kaggle'dan gelen "description" anahtarını en başa ekledik
    for k in ["description", "text", "raw_description", "body"]:
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None

def normalize(rec: Dict, source_hint: str) -> Optional[Dict]:
    text = pick_text(rec)
    if not text: return None

    return {
        "source": rec.get("source") or source_hint,
        "ref": rec.get("ref") or rec.get("id") or rec.get("name"),
        "title": rec.get("title") or "Untitled Dataset",
        "text": text,
        "url": rec.get("url"),
    }

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    total_out = 0

    with OUT.open("w", encoding="utf-8", newline="\n") as out:
        # Önce Hugging Face sonra Kaggle
        for path, hint in [(HF_IN, "huggingface"), (KG_IN, "kaggle")]:
            print(f"[KONTROL] {path} dosyasına bakılıyor...")
            for rec in iter_jsonl(path):
                norm = normalize(rec, hint)
                if norm:
                    out.write(json.dumps(norm, ensure_ascii=False) + "\n")
                    total_out += 1

    print(f"\n[BAŞARILI] Toplam {total_out} veri birleştirildi.")
    print(f"[DOSYA] {OUT}")

if __name__ == "__main__":
    main()