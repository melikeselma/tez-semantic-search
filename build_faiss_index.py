import json
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent

# Yolları search.py ile uyumlu hale getirdik
IN_PATH = ROOT / "data" / "processed" / "descriptions_clean.jsonl"
IDX_DIR = ROOT / "data" / "index"

# Klasörü oluştur
IDX_DIR.mkdir(parents=True, exist_ok=True)

# search.py'nin beklediği dosya isimleri
OUT_INDEX = IDX_DIR / "faiss.index"
OUT_MAPPING = IDX_DIR / "mappings.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 32

def iter_jsonl(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"{path} bulunamadı! Önce normalize_merge.py çalıştır.")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def main():
    rows = list(iter_jsonl(IN_PATH))
    texts = []
    # search.py 'mappings.json' içinde bir sözlük (dict) bekliyor
    mapping_dict = {}

    for i, r in enumerate(rows):
        t = r.get("text")
        if isinstance(t, str) and t.strip():
            texts.append(t.strip())
            # search.py'nin beklediği formatta veri saklıyoruz
            mapping_dict[i] = {
                "source": r.get("source"),
                "ref": r.get("ref"),
                "title": r.get("title"),
                "url": r.get("url"),
                "text": t.strip() # Özet gösterimi için gerekli
            }

    if not texts:
        raise RuntimeError("İşlenecek metin bulunamadı!")

    print(f"[LOAD] {len(texts)} metin vektöre çevriliyor...")

    model = SentenceTransformer(MODEL_NAME)
    emb = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype("float32")

    # 1. FAISS İndeksini oluştur ve kaydet
    d = emb.shape[1]
    index = faiss.IndexFlatIP(d) # İç çarpım (Cosine similarity benzeri)
    index.add(emb)
    faiss.write_index(index, str(OUT_INDEX))

    # 2. Mappings dosyasını search.py'nin istediği isimle ve formatta kaydet
    with OUT_MAPPING.open("w", encoding="utf-8") as f:
        json.dump(mapping_dict, f, ensure_ascii=False)

    print(f"\n[BAŞARILI]")
    print(f"-> İndeks: {OUT_INDEX}")
    print(f"-> Rehber: {OUT_MAPPING}")
    print(f"Toplam {index.ntotal} veri indekslendi.")

if __name__ == "__main__":
    main()