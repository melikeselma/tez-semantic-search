import json
import random
import time
from pathlib import Path
from datetime import datetime, timezone

import requests
from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = Path(r"C:\Users\user\Desktop\Tez")
REFS_PATH = ROOT / "data" / "raw" / "kaggle" / "kaggle_refs.jsonl"
OUT_PATH = ROOT / "data" / "raw" / "kaggle" / "raw_kaggle.jsonl"
TMP_META = ROOT / "data" / "raw" / "kaggle" / "_tmp_meta"

TARGET = 800
SLEEP_EACH = 6.0          # Kaggle 429 çok sert -> biraz yükselttim
JITTER = 4.0
PROGRESS_EVERY = 25


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def kaggle_call(fn, *args, **kwargs):
    max_tries = 12
    for attempt in range(max_tries):
        try:
            return fn(*args, **kwargs)
        except requests.exceptions.HTTPError as e:
            resp = getattr(e, "response", None)
            status = getattr(resp, "status_code", None)

            if status == 429 or (status is not None and 500 <= status <= 599):
                wait_s = min(900, (60 * (2 ** attempt))) + random.random() * 10
                print(f"[RATE LIMIT] status={status} wait={wait_s:.1f}s attempt={attempt+1}/{max_tries}")
                time.sleep(wait_s)
                continue
            raise
        except Exception as e:
            msg = str(e).lower()
            if "too many requests" in msg or "429" in msg:
                wait_s = min(900, (60 * (2 ** attempt))) + random.random() * 10
                print(f"[RATE LIMIT] (generic) wait={wait_s:.1f}s attempt={attempt+1}/{max_tries}")
                time.sleep(wait_s)
                continue
            raise
    raise RuntimeError("Kaggle call failed after retries")


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_done_refs(path: Path):
    done = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                r = obj.get("ref")
                if r:
                    done.add(r)
            except Exception:
                continue
    return done


def safe_load_json_maybe_double(path: Path):
    """
    dataset-metadata.json Kaggle tarafından bazen JSON objesi yerine JSON-escaped string olarak yazılıyor.
    Bu fonksiyon:
      1) json.loads -> dict ise döndürür
      2) json.loads -> str ise, ikinci kez json.loads dener
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            return None
        obj = json.loads(text)

        # Normal case: dict
        if isinstance(obj, dict):
            return obj

        # Kaggle weird case: JSON string inside JSON
        if isinstance(obj, str):
            obj2 = json.loads(obj)
            if isinstance(obj2, dict):
                return obj2

        return None
    except Exception:
        return None


def load_meta_dict(ds_dir: Path):
    candidates = [
        ds_dir / "dataset-metadata.json",
        ds_dir / "datapackage.json",
        ds_dir / "dataset_metadata.json",
    ]
    for p in candidates:
        if p.exists():
            d = safe_load_json_maybe_double(p)
            if d is not None:
                return d
    return None


def extract_description(meta: dict):
    # Kaggle örneğinde description direkt var
    for k in ["description", "subtitle", "summary", "overview"]:
        v = meta.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    TMP_META.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    done_refs = load_done_refs(OUT_PATH)
    collected = len(done_refs)
    print(f"[START] already_collected={collected}, target={TARGET}")

    with OUT_PATH.open("a", encoding="utf-8", newline="\n") as out:
        for rec in read_jsonl(REFS_PATH):
            if collected >= TARGET:
                break

            ref = rec.get("ref")
            title = rec.get("title")
            if not ref or ref in done_refs:
                continue

            safe_ref = ref.replace("/", "__")
            ds_dir = TMP_META / safe_ref
            ds_dir.mkdir(parents=True, exist_ok=True)

            # metadata indir
            kaggle_call(api.dataset_metadata, ref, path=str(ds_dir))

            meta = load_meta_dict(ds_dir)
            if not meta:
                continue

            desc = extract_description(meta)
            if not desc:
                continue

            out.write(json.dumps({
                "source": "kaggle",
                "ref": ref,
                "title": title,
                "description": desc,
                "url": f"https://www.kaggle.com/datasets/{ref}",
                "collected_at": now_iso(),
            }, ensure_ascii=False) + "\n")
            out.flush()

            done_refs.add(ref)
            collected += 1

            if collected % PROGRESS_EVERY == 0:
                print(f"[OK] {collected}/{TARGET} description yazıldı.")

            time.sleep(SLEEP_EACH + random.random() * JITTER)

    print(f"[DONE] Kaggle raw: {OUT_PATH}")
    print(f"[DONE] count: {collected}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Ctrl+C alındı. Yazılan kayıtlar dosyada duruyor (append+flush aktif).")