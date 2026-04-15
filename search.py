import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import os

# --- YOLLARI GARANTİYE ALALIM ---
BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = BASE_DIR / "data" / "index" / "faiss.index"
MAPPING_PATH = BASE_DIR / "data" / "index" / "mappings.json"

def main():
    # 1. DOSYA KONTROLÜ (Nereye bakıyoruz?)
    if not INDEX_PATH.exists():
        # Eğer yukarıdaki bulamazsa alternatif bir yol daha dene (Monster kullanıcısı için özel)
        INDEX_PATH_ALT = Path(r"C:\Users\MONSTER\Desktop\tez-semantic-search\data\index\faiss.index")
        MAPPING_PATH_ALT = Path(r"C:\Users\MONSTER\Desktop\tez-semantic-search\data\index\mappings.json")
        
        if INDEX_PATH_ALT.exists():
            print(f"[LOG] Dosyalar alternatif yolda bulundu.")
            current_index_path = INDEX_PATH_ALT
            current_mapping_path = MAPPING_PATH_ALT
        else:
            print(f"\n[HATA] İndeks dosyası bulunamadı!")
            print(f"Baktığım yer 1: {INDEX_PATH.absolute()}")
            print(f"Baktığım yer 2: {INDEX_PATH_ALT}")
            print("\nLütfen 'data/index' klasöründe 'faiss.index' olduğundan emin ol.")
            return
    else:
        current_index_path = INDEX_PATH
        current_mapping_path = MAPPING_PATH

    # 2. YÜKLEME AŞAMASI
    print("[LOG] Yapay zeka modeli (MiniLM) yükleniyor...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("[LOG] Vektör veritabanı yükleniyor...")
    index = faiss.read_index(str(current_index_path))
    
    with open(current_mapping_path, "r", encoding="utf-8") as f:
        mappings = json.load(f)

    # 3. KULLANICI ARAYÜZÜ
    print("\n" + "="*50)
    print("   SEMANTİK VERİ SETİ ARAMA MOTORUNA HOŞ GELDİNİZ")
    print("="*50)

    while True:
        query = input("\nNe tür bir veri seti arıyorsunuz? (Çıkış için 'q'): ").strip()
        if query.lower() == 'q': break
        if not query: continue

        # Sorguyu vektöre çevir
        query_vec = model.encode([query]).astype("float32")
        
        # En yakın 5 sonucu getir
        distances, indices = index.search(query_vec, k=5)

        print(f"\n'{query}' için en alakalı sonuçlar:")
        print("-" * 50)

        for i, idx in enumerate(indices[0]):
            if idx == -1: continue
            
            res = mappings.get(str(idx))
            if not res: continue

            print(f"{i+1}. BAŞLIK: {res.get('title', 'İsimsiz')}")
            print(f"   KAYNAK: {res.get('source', 'Kaggle/HF').upper()}")
            print(f"   URL: {res.get('url', 'N/A')}")
            
            # Metnin ilk kısmını temizce göster
            desc = res.get('text', '').replace('\n', ' ')
            print(f"   ÖZET: {desc[:150]}...")
            print("-" * 50)

if __name__ == "__main__":
    main()