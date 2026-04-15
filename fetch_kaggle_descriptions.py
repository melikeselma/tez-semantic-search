import json
import os
import time
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = Path(__file__).resolve().parent
REFS_PATH = ROOT / "data" / "raw" / "kaggle" / "kaggle_refs.jsonl"
OUT_PATH = ROOT / "data" / "raw" / "kaggle" / "raw_kaggle.jsonl"
TMP_META = ROOT / "temp_kg" 

def deep_find_description(obj):
    """JSON içinde nerede açıklama varsa bulur."""
    # 1. Klasik anahtarlar
    for key in ["description", "subtitle", "summary", "text"]:
        if obj.get(key) and isinstance(obj[key], str) and len(obj[key]) > 10:
            return obj[key]
    
    # 2. userMetadata altındaki yerler
    user_meta = obj.get("userMetadata", {})
    if isinstance(user_meta, dict):
        for key in ["description", "text"]:
            if user_meta.get(key) and len(str(user_meta[key])) > 10:
                return user_meta[key]
                
    # 3. Son çare: JSON içindeki en uzun metin bloğunu bul (Genelde açıklamadır)
    best_text = ""
    def search_dict(d):
        nonlocal best_text
        if isinstance(d, dict):
            for v in d.values(): search_dict(v)
        elif isinstance(d, list):
            for v in d: search_dict(v)
        elif isinstance(d, str):
            if len(d) > len(best_text):
                best_text = d
    
    search_dict(obj)
    return best_text if len(best_text) > 20 else None

def main():
    TMP_META.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()

    with open(REFS_PATH, "r", encoding="utf-8") as f:
        refs = [json.loads(line) for line in f]

    count = 0
    target = 100 

    with open(OUT_PATH, "a", encoding="utf-8") as out_f:
        for rec in refs:
            if count >= target: break
            ref = rec["ref"]
            print(f"[DENENİYOR] {ref}", end=" ")
            
            try:
                safe_name = ref.replace("/", "__")
                ds_dir = TMP_META / safe_name
                ds_dir.mkdir(parents=True, exist_ok=True)
                
                # API çağrısı
                api.dataset_metadata(ref, path=str(ds_dir))
                
                meta_file = ds_dir / "dataset-metadata.json"
                if meta_file.exists():
                    with open(meta_file, "r", encoding="utf-8") as m:
                        meta_data = json.load(m)
                        if isinstance(meta_data, str): meta_data = json.loads(meta_data)
                    
                    desc = deep_find_description(meta_data)
                    
                    if desc:
                        # Gereksiz HTML'leri çok basitçe temizle
                        desc_clean = desc.replace("<br>", "\n").replace("<b>", "").replace("</b>", "")
                        
                        out_f.write(json.dumps({
                            "source": "kaggle",
                            "ref": ref,
                            "title": rec.get("title"),
                            "description": desc_clean[:2000] # Çok uzunsa kes
                        }, ensure_ascii=False) + "\n")
                        out_f.flush()
                        count += 1
                        print(f"-> [TAMAM] {count}/{target}")
                    else:
                        print("-> [BOŞ]")
                else:
                    print("-> [DOSYA YOK]")
                
                time.sleep(6) # Kaggle ban koruması

            except Exception as e:
                print(f"-> [HATA] {e}")
                time.sleep(2)

if __name__ == "__main__":
    main()