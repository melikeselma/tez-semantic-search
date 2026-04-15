import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Dosya yollarını mevcut yapına göre ayarlıyoruz 
ROOT = Path(r"C:\Users\user\Desktop\Tez")
EMB_DIR = ROOT / "data" / "embeddings"
IDX_DIR = ROOT / "data" / "index"

META_PATH = EMB_DIR / "meta.jsonl"
INDEX_PATH = IDX_DIR / "faiss.index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" # build_faiss_index.py ile aynı model 

def load_search_engine():
    # 1. Meta verileri (başlık, url vb.) yükle
    meta = []
    with open(META_PATH, "r", encoding="utf-8") as f:
        for line in f:
            meta.append(json.loads(line))
    
    # 2. FAISS indeksini yükle
    index = faiss.read_index(str(INDEX_PATH))
    
    # 3. Modeli yükle
    model = SentenceTransformer(MODEL_NAME)
    
    return meta, index, model

def search(query, meta, index, model, k=5):
    # Sorguyu vektöre çevir ve normalize et 
    query_vector = model.encode([query], normalize_embeddings=True).astype("float32")
    
    # İndekste ara
    distances, indices = index.search(query_vector, k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(meta):
            res = meta[idx].copy()
            res["score"] = float(distances[0][i])
            results.append(res)
    return results

if __name__ == "__main__":
    print("Sistem yükleniyor...")
    meta, index, model = load_search_engine()
    
    while True:
        query = input("\nAramak istediğiniz konuyu yazın (Çıkış için 'q'): ")
        if query.lower() == 'q':
            break
            
        results = search(query, meta, index, model)
        
        print(f"\n'{query}' için en yakın sonuçlar:")
        for r in results:
            print(f"- {r['title']} (Skor: {r['score']:.4f})")
            print(f"  Link: {r['url']}\n")
        