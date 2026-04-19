import json
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from bm25 import BM25Index, search as bm25_search
from hybrid import DEFAULT_CANDIDATE_K, DEFAULT_SEMANTIC_WEIGHT, search as hybrid_search
from search import load_mappings, load_search_engine, search as semantic_search

BASE_DIR = Path(__file__).resolve().parent
INDEX_METADATA_PATH = BASE_DIR / "data" / "index" / "index_metadata.json"
HOST = "127.0.0.1"
PORT = 8000
MAX_BODY_BYTES = 16_384


HTML = """<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Semantik Veri Seti Arama</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f7f2;
      --ink: #151a17;
      --muted: #59645d;
      --line: #cfd8d1;
      --panel: #ffffff;
      --accent: #087f6f;
      --accent-strong: #066456;
      --warn: #b42342;
      --soft: #e7f2ee;
      --tag: #f3e9d2;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      background: var(--bg);
      color: var(--ink);
      letter-spacing: 0;
    }

    main {
      width: min(1120px, calc(100% - 32px));
      margin: 0 auto;
      padding: 28px 0 40px;
    }

    .topbar {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 18px;
      margin-bottom: 20px;
    }

    h1 {
      margin: 0 0 8px;
      font-size: 30px;
      line-height: 1.15;
    }

    .subtitle {
      margin: 0;
      color: var(--muted);
      font-size: 15px;
      line-height: 1.45;
      max-width: 680px;
    }

    .stats {
      display: flex;
      flex-wrap: wrap;
      justify-content: flex-end;
      gap: 8px;
      min-width: 260px;
    }

    .stat {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 8px 10px;
      min-width: 92px;
    }

    .stat strong {
      display: block;
      font-size: 18px;
      line-height: 1.15;
    }

    .stat span {
      color: var(--muted);
      font-size: 12px;
    }

    .search-panel {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 14px;
      margin-bottom: 18px;
    }

    form {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 150px 150px 120px 132px;
      gap: 10px;
      align-items: stretch;
    }

    label {
      display: block;
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 6px;
    }

    input,
    select,
    button {
      width: 100%;
      min-height: 44px;
      border-radius: 8px;
      font: inherit;
      letter-spacing: 0;
    }

    input,
    select {
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      padding: 10px 12px;
    }

    input:focus,
    select:focus {
      outline: 3px solid var(--soft);
      border-color: var(--accent);
    }

    button {
      border: 1px solid var(--accent-strong);
      background: var(--accent);
      color: #fff;
      cursor: pointer;
      font-weight: 700;
      padding: 0 14px;
      margin-top: 19px;
    }

    button:hover {
      background: var(--accent-strong);
    }

    button:disabled {
      cursor: wait;
      opacity: 0.7;
    }

    .examples {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 12px;
    }

    .example-btn,
    .detail-btn,
    .close-btn {
      width: auto;
      min-height: 34px;
      margin: 0;
      border-radius: 8px;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--accent-strong);
      padding: 6px 10px;
      font-size: 13px;
      font-weight: 700;
      cursor: pointer;
    }

    .example-btn:hover,
    .detail-btn:hover,
    .close-btn:hover {
      border-color: var(--accent);
      background: var(--soft);
    }

    .score-note {
      margin: 10px 0 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }

    .status {
      min-height: 22px;
      margin: 0 0 12px;
      color: var(--muted);
      font-size: 14px;
    }

    .status.error {
      color: var(--warn);
      font-weight: 700;
    }

    .results {
      display: grid;
      gap: 10px;
    }

    .result {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 14px;
    }

    .result-head {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 8px;
    }

    .result h2 {
      margin: 0;
      font-size: 19px;
      line-height: 1.25;
      overflow-wrap: anywhere;
    }

    .score {
      flex: 0 0 auto;
      border-radius: 8px;
      background: var(--soft);
      color: var(--accent-strong);
      padding: 6px 8px;
      font-weight: 700;
      font-size: 13px;
    }

    .meta {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 10px;
    }

    .chip {
      border-radius: 8px;
      background: var(--tag);
      color: #4e3d10;
      padding: 5px 8px;
      font-size: 13px;
    }

    .summary {
      margin: 0 0 10px;
      color: #2c332f;
      line-height: 1.5;
      overflow-wrap: anywhere;
    }

    a {
      color: var(--accent-strong);
      font-weight: 700;
      overflow-wrap: anywhere;
    }

    .actions {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
    }

    .empty {
      border: 1px dashed var(--line);
      border-radius: 8px;
      padding: 18px;
      color: var(--muted);
      background: rgba(255, 255, 255, 0.64);
    }

    .modal-backdrop {
      position: fixed;
      inset: 0;
      display: none;
      align-items: stretch;
      justify-content: flex-end;
      padding: 0;
      background: rgba(21, 26, 23, 0.48);
      z-index: 10;
    }

    .modal-backdrop.open {
      display: flex;
    }

    .modal {
      width: min(520px, 100%);
      height: 100vh;
      overflow: auto;
      border-radius: 8px 0 0 8px;
      background: #fff;
      border-left: 1px solid var(--line);
      padding: 18px;
      box-shadow: -12px 0 30px rgba(21, 26, 23, 0.18);
    }

    .modal-head {
      display: flex;
      justify-content: space-between;
      gap: 14px;
      align-items: flex-start;
      margin-bottom: 12px;
    }

    .modal h2 {
      margin: 0;
      font-size: 21px;
      line-height: 1.25;
      overflow-wrap: anywhere;
    }

    .modal dl {
      display: grid;
      grid-template-columns: 136px minmax(0, 1fr);
      gap: 8px 12px;
      margin: 0 0 12px;
    }

    .modal dt {
      color: var(--muted);
      font-weight: 700;
    }

    .modal dd {
      margin: 0;
      overflow-wrap: anywhere;
    }

    .modal-section {
      border-top: 1px solid var(--line);
      padding-top: 12px;
      margin-top: 12px;
    }

    .modal-section h3 {
      margin: 0 0 8px;
      font-size: 15px;
    }

    .why-box {
      border-radius: 8px;
      background: var(--soft);
      color: #153d36;
      padding: 10px;
      line-height: 1.45;
      font-size: 14px;
    }

    .modal-text {
      white-space: pre-wrap;
      line-height: 1.5;
      overflow-wrap: anywhere;
    }

    @media (max-width: 760px) {
      main {
        width: min(100% - 20px, 1120px);
        padding-top: 18px;
      }

      .topbar {
        display: block;
      }

      .stats {
        justify-content: flex-start;
        margin-top: 14px;
      }

      form {
        grid-template-columns: 1fr;
      }

      button {
        margin-top: 0;
      }

      h1 {
        font-size: 25px;
      }

      .result-head {
        display: block;
      }

      .score {
        display: inline-block;
        margin-top: 8px;
      }

      .modal dl {
        grid-template-columns: 1fr;
      }

      .modal {
        width: 100%;
        border-radius: 0;
      }
    }
  </style>
</head>
<body>
  <main>
    <div class="topbar">
      <div>
        <h1>Semantik Veri Seti Arama</h1>
        <p class="subtitle">Aradığınız konuyu doğal dille yazın; sistem Kaggle ve Hugging Face açıklamaları içinde anlamca en yakın veri setlerini sıralar.</p>
      </div>
      <div class="stats" id="stats"></div>
    </div>

    <section class="search-panel" aria-label="Arama">
      <form id="search-form">
        <div>
          <label for="query">Sorgu</label>
          <input id="query" name="query" autocomplete="off" placeholder="Örn. earthquake data, audio deepfake, movie reviews" required>
        </div>
        <div>
          <label for="method">Yöntem</label>
          <select id="method" name="method">
            <option value="semantic">Semantic</option>
            <option value="bm25">BM25</option>
            <option value="hybrid">Hybrid</option>
          </select>
        </div>
        <div>
          <label for="source">Kaynak</label>
          <select id="source" name="source">
            <option value="all">Tümü</option>
            <option value="kaggle">Kaggle</option>
            <option value="huggingface">Hugging Face</option>
          </select>
        </div>
        <div>
          <label for="top-k">Sonuç</label>
          <select id="top-k" name="top_k">
            <option value="5">Top 5</option>
            <option value="10">Top 10</option>
            <option value="15">Top 15</option>
          </select>
        </div>
        <div>
          <button id="submit" type="submit">Ara</button>
        </div>
      </form>
      <div class="examples" id="examples" aria-label="Örnek sorgular">
        <button class="example-btn" type="button" data-query="earthquake data">earthquake data</button>
        <button class="example-btn" type="button" data-query="audio deepfake">audio deepfake</button>
        <button class="example-btn" type="button" data-query="stock price history">stock price history</button>
        <button class="example-btn" type="button" data-query="movie reviews">movie reviews</button>
        <button class="example-btn" type="button" data-query="medical image classification">medical image classification</button>
        <button class="example-btn" type="button" data-query="weather dataset">weather dataset</button>
      </div>
      <p class="score-note">Semantic skoru embedding benzerliğidir; BM25 skoru kelime eşleşmesi ağırlığıdır. Hybrid, iki skoru normalize edip birleştirir. Skorlar farklı yöntemler arasında doğrudan karşılaştırılmaz.</p>
    </section>

    <p id="status" class="status">Hazır.</p>
    <section id="results" class="results" aria-live="polite">
      <div class="empty">Bir sorgu girildiğinde sonuçlar burada listelenir.</div>
    </section>
  </main>

  <div class="modal-backdrop" id="detail-modal" aria-hidden="true">
    <div class="modal" role="dialog" aria-modal="true" aria-labelledby="modal-title">
      <div class="modal-head">
        <h2 id="modal-title">Detay</h2>
        <button class="close-btn" id="modal-close" type="button">Kapat</button>
      </div>
      <dl id="modal-meta"></dl>
      <section class="modal-section" aria-label="Neden gösterildi">
        <h3>Neden bu dataset gösterildi?</h3>
        <div class="why-box" id="modal-why"></div>
      </section>
      <section class="modal-section" aria-label="Tam açıklama">
        <h3>Tam açıklama</h3>
        <div class="modal-text" id="modal-text"></div>
      </section>
    </div>
  </div>

  <script>
    const form = document.querySelector("#search-form");
    const queryInput = document.querySelector("#query");
    const methodInput = document.querySelector("#method");
    const sourceInput = document.querySelector("#source");
    const topKInput = document.querySelector("#top-k");
    const submitButton = document.querySelector("#submit");
    const statusEl = document.querySelector("#status");
    const resultsEl = document.querySelector("#results");
    const statsEl = document.querySelector("#stats");
    const examplesEl = document.querySelector("#examples");
    const detailModal = document.querySelector("#detail-modal");
    const modalTitle = document.querySelector("#modal-title");
    const modalMeta = document.querySelector("#modal-meta");
    const modalWhy = document.querySelector("#modal-why");
    const modalText = document.querySelector("#modal-text");
    const modalClose = document.querySelector("#modal-close");
    let currentResults = [];
    let currentMethod = "semantic";

    const escapeHtml = (value) => String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");

    const sourceLabels = {
      all: "Tüm kaynaklar",
      kaggle: "Kaggle",
      huggingface: "Hugging Face"
    };

    function scoreLabel(method) {
      if (method === "bm25") return "BM25";
      if (method === "hybrid") return "HYB";
      return "SEM";
    }

    function formatScore(score) {
      const numeric = Number(score);
      return Number.isFinite(numeric) ? numeric.toFixed(4) : "-";
    }

    function formatList(value, emptyLabel = "-") {
      if (Array.isArray(value)) {
        return value.length ? value.join(", ") : emptyLabel;
      }
      return value || emptyLabel;
    }

    function similarityExplanation(method, score) {
      const formattedScore = formatScore(score);
      if (method === "hybrid") {
        return `Bu sonuç, Semantic ve BM25 sonuçları birlikte değerlendirildiği için listelendi. Hybrid skor ${formattedScore}; sistem anlam benzerliği skorunu ve kelime eşleşmesi skorunu normalize edip ağırlıklı olarak birleştirir.`;
      }
      if (method === "bm25") {
        return `Bu sonuç, sorgudaki kelimeler dataset başlığı ve açıklamasında güçlü biçimde geçtiği için listelendi. BM25 skoru ${formattedScore}; bu skor kelime eşleşme sıklığına, kelimenin ayırt ediciliğine ve açıklama uzunluğuna göre hesaplanır.`;
      }
      return `Bu sonuç, sorgunun embedding vektörü ile dataset açıklamasının embedding vektörü birbirine yakın olduğu için listelendi. Benzerlik skoru ${formattedScore}; 1'e yaklaştıkça anlamsal yakınlık artar.`;
    }

    function renderStats(metadata) {
      const sources = metadata.source_counts || {};
      const items = [
        ["Toplam", metadata.indexed_rows ?? "-"],
        ["Kaggle", sources.kaggle ?? "-"],
        ["Hugging Face", sources.huggingface ?? "-"]
      ];
      statsEl.innerHTML = items.map(([label, value]) => `
        <div class="stat"><strong>${escapeHtml(value)}</strong><span>${escapeHtml(label)}</span></div>
      `).join("");
    }

    async function loadMetadata() {
      try {
        const response = await fetch("/api/metadata");
        if (!response.ok) return;
        renderStats(await response.json());
      } catch {
        statsEl.innerHTML = "";
      }
    }

    function renderResults(results, method) {
      currentResults = results;
      currentMethod = method;

      if (!results.length) {
        resultsEl.innerHTML = '<div class="empty">Bu sorgu için sonuç bulunamadı.</div>';
        return;
      }

      resultsEl.innerHTML = results.map((result, index) => {
        const text = String(result.text || "");
        const summary = text.length > 360 ? `${text.slice(0, 360)}...` : text;
        const url = result.url ? `<a href="${escapeHtml(result.url)}" target="_blank" rel="noreferrer">Kaynağa git</a>` : "URL yok";
        return `
          <article class="result">
            <div class="result-head">
              <h2>${index + 1}. ${escapeHtml(result.title || "İsimsiz veri seti")}</h2>
              <span class="score">${scoreLabel(method)} ${formatScore(result.score)}</span>
            </div>
            <div class="meta">
              <span class="chip">${escapeHtml((result.source || "unknown").toUpperCase())}</span>
              <span class="chip">${escapeHtml(result.ref || "ref yok")}</span>
            </div>
            <p class="summary">${escapeHtml(summary)}</p>
            <div class="actions">
              <button class="detail-btn" type="button" data-result-index="${index}">Detay</button>
              ${url}
            </div>
          </article>
        `;
      }).join("");
    }

    function openDetail(index) {
      const result = currentResults[index];
      if (!result) return;

      modalTitle.textContent = result.title || "İsimsiz veri seti";
      const urlValue = result.url
        ? `<a href="${escapeHtml(result.url)}" target="_blank" rel="noreferrer">${escapeHtml(result.url)}</a>`
        : "URL yok";
      const metaItems = [
        ["Başlık", result.title || "İsimsiz veri seti"],
        ["Yöntem", scoreLabel(currentMethod)],
        ["Skor", formatScore(result.score)],
        ["Kaynak", (result.source || "unknown").toUpperCase()],
        ["Referans", result.ref || "ref yok"],
        ["URL", urlValue],
        ["Keywords", formatList(result.keywords)],
        ["Licence", result.license || "-"],
        ["Description length", `${result.description_len_chars ?? "-"} karakter / ${result.description_len_words ?? "-"} kelime`],
        ["Quality flag", formatList(result.quality_flags, "flag yok")]
      ];
      if (currentMethod === "hybrid") {
        metaItems.push(["Semantic bileşen", formatScore(result.semantic_score)]);
        metaItems.push(["BM25 bileşen", formatScore(result.bm25_score)]);
        metaItems.push(["Ağırlıklar", `Semantic ${formatScore(result.semantic_weight)} / BM25 ${formatScore(result.bm25_weight)}`]);
      }
      modalMeta.innerHTML = metaItems.map(([label, value]) => `
        <dt>${escapeHtml(label)}</dt><dd>${label === "URL" ? value : escapeHtml(value)}</dd>
      `).join("");
      modalWhy.textContent = similarityExplanation(currentMethod, result.score);
      modalText.textContent = result.description || result.text || "";
      detailModal.classList.add("open");
      detailModal.setAttribute("aria-hidden", "false");
      modalClose.focus();
    }

    function closeDetail() {
      detailModal.classList.remove("open");
      detailModal.setAttribute("aria-hidden", "true");
    }

    resultsEl.addEventListener("click", (event) => {
      const button = event.target.closest(".detail-btn");
      if (!button) return;
      openDetail(Number(button.dataset.resultIndex));
    });

    modalClose.addEventListener("click", closeDetail);
    detailModal.addEventListener("click", (event) => {
      if (event.target === detailModal) closeDetail();
    });
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") closeDetail();
    });

    examplesEl.addEventListener("click", (event) => {
      const button = event.target.closest(".example-btn");
      if (!button) return;
      queryInput.value = button.dataset.query || "";
      form.requestSubmit();
    });

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const query = queryInput.value.trim();
      const method = methodInput.value;
      const source = sourceInput.value;
      const topK = Number(topKInput.value);
      if (!query) return;

      submitButton.disabled = true;
      statusEl.className = "status";
      statusEl.textContent = "Aranıyor...";

      try {
        const response = await fetch("/api/search", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query, method, source, top_k: topK })
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || "Arama tamamlanamadı.");

        renderResults(payload.results || [], payload.method);
        const sourceLabel = sourceLabels[payload.source] || "Tüm kaynaklar";
        statusEl.textContent = `${payload.method.toUpperCase()} / ${sourceLabel} ile "${payload.query}" için ${payload.results.length} sonuç listelendi.`;
      } catch (error) {
        statusEl.className = "status error";
        statusEl.textContent = error.message;
      } finally {
        submitButton.disabled = false;
      }
    });

    loadMetadata();
    queryInput.focus();
  </script>
</body>
</html>
"""


def configure_output():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def load_index_metadata():
    if not INDEX_METADATA_PATH.exists():
        return {}
    with INDEX_METADATA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


class AppState:
    model = None
    index = None
    mappings = None
    bm25_index = None
    metadata = None
    lock = threading.Lock()


def ensure_search_engine():
    if AppState.model is not None:
        return

    with AppState.lock:
        if AppState.model is not None:
            return
        print("[WEB] Arama motoru yükleniyor...")
        AppState.model, AppState.index, AppState.mappings = load_search_engine()


def ensure_bm25_engine():
    if AppState.bm25_index is not None:
        return

    with AppState.lock:
        if AppState.bm25_index is not None:
            return
        if AppState.mappings is None:
            AppState.mappings = load_mappings()
        print("[WEB] BM25 index oluşturuluyor...")
        AppState.bm25_index = BM25Index(AppState.mappings)


def filter_results_by_source(raw_results, source_filter, top_k):
    results = []
    for score, item in raw_results:
        if source_filter != "all" and (item.get("source") or "").lower() != source_filter:
            continue
        results.append((score, item))
        if len(results) >= top_k:
            break
    return results


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "SemanticDatasetSearch/1.0"

    def log_message(self, format, *args):
        print(f"[WEB] {self.address_string()} - {format % args}")

    def send_json(self, payload, status=HTTPStatus.OK):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_html(self):
        body = HTML.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self.send_html()
            return
        if path == "/api/metadata":
            self.send_json(AppState.metadata or {})
            return
        self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self):
        path = urlparse(self.path).path
        if path != "/api/search":
            self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_json({"error": "Invalid Content-Length"}, HTTPStatus.BAD_REQUEST)
            return

        if content_length <= 0 or content_length > MAX_BODY_BYTES:
            self.send_json({"error": "Invalid request body"}, HTTPStatus.BAD_REQUEST)
            return

        try:
            body = self.rfile.read(content_length).decode("utf-8")
            payload = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.send_json({"error": "Invalid JSON"}, HTTPStatus.BAD_REQUEST)
            return

        query = str(payload.get("query", "")).strip()
        if not query:
            self.send_json({"error": "Sorgu boş olamaz."}, HTTPStatus.BAD_REQUEST)
            return

        method = str(payload.get("method", "semantic")).strip().lower()
        if method not in {"semantic", "bm25", "hybrid"}:
            self.send_json({"error": "Unknown search method"}, HTTPStatus.BAD_REQUEST)
            return

        source_filter = str(payload.get("source", "all")).strip().lower()
        if source_filter not in {"all", "kaggle", "huggingface"}:
            self.send_json({"error": "Unknown source filter"}, HTTPStatus.BAD_REQUEST)
            return

        try:
            top_k = int(payload.get("top_k", 5))
        except (TypeError, ValueError):
            top_k = 5
        top_k = max(1, min(top_k, 20))

        try:
            if method == "bm25":
                ensure_bm25_engine()
                candidate_k = (
                    len(AppState.bm25_index.doc_ids)
                    if source_filter != "all"
                    else top_k
                )
                raw_results = bm25_search(query, AppState.bm25_index, top_k=candidate_k)
            elif method == "hybrid":
                ensure_search_engine()
                ensure_bm25_engine()
                candidate_k = (
                    len(AppState.mappings)
                    if source_filter != "all"
                    else max(top_k, min(DEFAULT_CANDIDATE_K, len(AppState.mappings)))
                )
                raw_results = hybrid_search(
                    query,
                    AppState.model,
                    AppState.index,
                    AppState.mappings,
                    AppState.bm25_index,
                    top_k=candidate_k if source_filter != "all" else top_k,
                    semantic_weight=DEFAULT_SEMANTIC_WEIGHT,
                    candidate_k=candidate_k,
                )
            else:
                ensure_search_engine()
                candidate_k = AppState.index.ntotal if source_filter != "all" else top_k
                raw_results = semantic_search(
                    query,
                    AppState.model,
                    AppState.index,
                    AppState.mappings,
                    top_k=candidate_k,
                )
        except Exception as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        filtered_results = filter_results_by_source(raw_results, source_filter, top_k)
        results = []
        for score, item in filtered_results:
            results.append(
                {
                    "score": score,
                    "source": item.get("source"),
                    "ref": item.get("ref"),
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "description": item.get("description"),
                    "text": item.get("text"),
                    "keywords": item.get("keywords") or [],
                    "license": item.get("license") or "",
                    "quality_flags": item.get("quality_flags") or [],
                    "description_len_chars": item.get("description_len_chars"),
                    "description_len_words": item.get("description_len_words"),
                    "semantic_score": item.get("semantic_score"),
                    "bm25_score": item.get("bm25_score"),
                    "semantic_weight": item.get("semantic_weight"),
                    "bm25_weight": item.get("bm25_weight"),
                }
            )

        self.send_json(
            {
                "query": query,
                "method": method,
                "source": source_filter,
                "top_k": top_k,
                "results": results,
            }
        )


def main():
    configure_output()
    AppState.metadata = load_index_metadata()

    server = ThreadingHTTPServer((HOST, PORT), RequestHandler)
    print(f"[WEB] UI hazır: http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[WEB] Kapatılıyor...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
