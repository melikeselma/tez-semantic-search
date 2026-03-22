import json
import random
import time
from pathlib import Path

import requests
from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = Path(r"C:\Users\user\Desktop\Tez")
OUT_PATH = ROOT / "data" / "raw" / "kaggle" / "kaggle_refs.jsonl"

TARGET_REFS = 2000
PAGE_SIZE = 10

# hız: 429 yemiyorsan bile nazik olalım
SLEEP_BASE = 1.2
SLEEP_JITTER = 1.0


def kaggle_call(fn, *args, **kwargs):
    max_tries = 12
    for attempt in range(max_tries):
        try:
            return fn(*args, **kwargs)
        except requests.exceptions.HTTPError as e:
            resp = getattr(e, "response", None)
            status = getattr(resp, "status_code", None)

            if status == 404:
                raise

            if status == 429 or (status is not None and 500 <= status <= 599):
                wait_s = min(600, (30 * (2 ** attempt))) + random.random() * 5
                print(f"[RATE LIMIT] status={status} wait={wait_s:.1f}s attempt={attempt+1}/{max_tries}")
                time.sleep(wait_s)
                continue
            raise
        except Exception as e:
            msg = str(e).lower()
            if "too many requests" in msg or "429" in msg:
                wait_s = min(600, (30 * (2 ** attempt))) + random.random() * 5
                print(f"[RATE LIMIT] (generic) wait={wait_s:.1f}s attempt={attempt+1}/{max_tries}")
                time.sleep(wait_s)
                continue
            raise
    raise RuntimeError("Kaggle call failed after retries")


def load_existing_refs(path: Path):
    seen = set()
    if not path.exists():
        return seen
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                r = obj.get("ref")
                if r:
                    seen.add(r)
            except Exception:
                continue
    return seen


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    seen = load_existing_refs(OUT_PATH)
    written = len(seen)

    # Sayfayı otomatik tahmin et (tam garanti değil ama yeterince iyi)
    page = max(1, (written // PAGE_SIZE) + 1)

    print(f"[START] existing_refs={written}, start_page≈{page}")

    with OUT_PATH.open("a", encoding="utf-8", newline="\n") as out:
        while written < TARGET_REFS:
            try:
                items = kaggle_call(api.dataset_list, page=page, max_size=PAGE_SIZE)
            except requests.exceptions.HTTPError as e:
                if getattr(e.response, "status_code", None) == 404:
                    print(f"[STOP] page={page} 404 -> liste bitti.")
                    break
                raise

            if not items:
                print(f"[STOP] page={page} boş liste.")
                break

            new_in_page = 0
            for it in items:
                ref = getattr(it, "ref", None)
                title = getattr(it, "title", None)
                if not ref or ref in seen:
                    continue

                seen.add(ref)
                out.write(json.dumps({"ref": ref, "title": title}, ensure_ascii=False) + "\n")
                written += 1
                new_in_page += 1

                if written % 200 == 0:
                    print(f"[OK] refs: {written}/{TARGET_REFS} (page={page})")

                if written >= TARGET_REFS:
                    break

            page += 1
            # sayfa hiç yeni üretmediyse de ilerleyelim ama nazikçe
            time.sleep(SLEEP_BASE + random.random() * SLEEP_JITTER)

    print(f"[DONE] refs path: {OUT_PATH}")
    print(f"[DONE] total refs: {written}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Ctrl+C alındı. Şu ana kadar yazılan refs dosyada duruyor (kayıp yok).")