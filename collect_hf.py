import time
import json
import uuid
import requests
from datetime import datetime, timezone

def utc_now():
    return datetime.now(timezone.utc).isoformat()

def list_datasets(limit=600):
    r = requests.get("https://huggingface.co/api/datasets", params={"limit": limit})
    r.raise_for_status()
    return r.json()

def fetch_readme(dataset_id):
    # 1) main dene
    for branch in ["main", "master"]:
        url = f"https://huggingface.co/datasets/{dataset_id}/raw/{branch}/README.md"
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200 and r.text.strip():
                return r.text
        except:
            pass
    return ""

def clean_readme(text):
    if not text:
        return ""
    # ilk 20 satırı al (genelde açıklama orada olur)
    lines = text.split("\n")
    short = "\n".join(lines[:20])
    return short.strip()

def main():
    output_path = "data/raw/huggingface/raw_hf.jsonl"
    datasets = list_datasets(limit=1500)

    with open(output_path, "w", encoding="utf-8") as f:
        for d in datasets:
            ds_id = d.get("id")
            if not ds_id:
                continue

            readme = fetch_readme(ds_id)
            raw_desc = clean_readme(readme)

            if not raw_desc:
                continue

            rec = {
                "id": str(uuid.uuid4()),
                "source": "huggingface",
                "source_id": ds_id,
                "title": ds_id,
                "raw_description": raw_desc,
                "url": f"https://huggingface.co/datasets/{ds_id}",
                "tags": d.get("tags") or [],
                "license": "",
                "language": "",
                "collected_at": utc_now(),
            }

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            time.sleep(0.2)

    print("Gerçek dataset açıklamaları çekildi.")

if __name__ == "__main__":
    main()