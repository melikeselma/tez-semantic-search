import json
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from bm25 import BM25Index, search as bm25_search
from hybrid import DEFAULT_CANDIDATE_K, DEFAULT_SEMANTIC_WEIGHT, search as hybrid_search
from query_understanding import build_query_plan
from search import load_mappings, load_search_engine, search as semantic_search
from search_profiles import DEFAULT_PROFILE_KEY, get_profile, get_profile_paths, list_profiles

BASE_DIR = Path(__file__).resolve().parent
HOST = "127.0.0.1"
PORT = 8000
MAX_BODY_BYTES = 16_384
DEFAULT_WEB_PROFILE = DEFAULT_PROFILE_KEY
RQ1_REPORT_DIR = BASE_DIR / "reports" / "evaluation" / "rq1_method_comparison"
RQ1_SUMMARY_PATH = RQ1_REPORT_DIR / "evaluation_summary.json"
RQ1_QUERY_STYLE_PATH = RQ1_REPORT_DIR / "evaluation_by_query_style.json"
RQ1_BENCHMARK_PATH = RQ1_REPORT_DIR / "evaluation_by_benchmark.json"
METHOD_LABELS = {
    "semantic": "Semantic",
    "bm25": "BM25",
    "hybrid": "Hybrid",
}
METHOD_ROLES = {
    "semantic": "Aciklama tabanli semantic retrieval",
    "bm25": "Lexical baseline",
    "hybrid": "En guclu pratik sistem",
}


HTML = """<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Semantic Dataset Search</title>
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
      max-width: 720px;
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
      min-width: 104px;
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

    .panel {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 14px;
      margin-bottom: 18px;
    }

    .panel h2 {
      margin: 0 0 8px;
      font-size: 20px;
    }

    .panel-note {
      margin: 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }

    form {
      display: grid;
      grid-template-columns: minmax(0, 1.6fr) 170px 150px 150px 120px 120px;
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

    .score-note,
    .status {
      margin: 10px 0 0;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.45;
    }

    .status {
      min-height: 22px;
      margin-bottom: 12px;
    }

    .status.error {
      color: var(--warn);
      font-weight: 700;
    }

    .research-list {
      display: grid;
      gap: 10px;
      margin: 0;
      padding: 0;
      list-style: none;
    }

    .research-item {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fbfcfb;
    }

    .research-item strong {
      display: block;
      margin-bottom: 6px;
    }

    .research-item p {
      margin: 0;
      color: #2c332f;
      line-height: 1.45;
    }

    .rq1-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin-top: 12px;
    }

    .rq1-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfb;
      padding: 12px;
    }

    .rq1-card h3,
    .rq1-card h4 {
      margin: 0 0 6px;
      font-size: 16px;
    }

    .rq1-label {
      display: inline-block;
      margin-bottom: 8px;
      color: var(--accent-strong);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.02em;
      text-transform: uppercase;
    }

    .rq1-card p {
      margin: 0 0 8px;
      color: #2c332f;
      line-height: 1.45;
      font-size: 14px;
    }

    .rq1-metrics {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }

    .rq1-metric {
      min-width: 86px;
      padding: 7px 8px;
      border-radius: 8px;
      background: var(--soft);
    }

    .rq1-metric strong {
      display: block;
      font-size: 15px;
    }

    .rq1-metric span {
      font-size: 12px;
      color: var(--muted);
    }

    .rq1-banner {
      margin-top: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfb;
      padding: 12px;
    }

    .rq1-banner strong {
      display: block;
      margin-bottom: 6px;
    }

    .rq1-banner p {
      margin: 0;
      line-height: 1.5;
      color: #2c332f;
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
      display: inline-block;
      border-radius: 8px;
      padding: 5px 8px;
      font-size: 12px;
      font-weight: 700;
      background: var(--tag);
      color: #4e3d10;
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
      background: rgba(21, 26, 23, 0.48);
      z-index: 10;
    }

    .modal-backdrop.open {
      display: flex;
    }

    .modal {
      width: min(560px, 100%);
      height: 100vh;
      overflow: auto;
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

    @media (max-width: 920px) {
      form {
        grid-template-columns: 1fr;
      }

      .rq1-grid {
        grid-template-columns: 1fr;
      }
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
      }
    }
  </style>
</head>
<body>
  <main>
    <div class="topbar">
      <div>
        <h1>Semantic Dataset Search</h1>
        <p class="subtitle">Bu arayuz, tezdeki semantic data discovery problemini canli olarak test etmek icin kullanilir. Aramada model, method ve kaynak degistirilebilir.</p>
      </div>
      <div class="stats" id="stats"></div>
    </div>

    <section class="panel" aria-label="Search">
      <h2>Live Search</h2>
        <p class="panel-note">Ayni sorguda tezdeki ana Ingilizce profil ile EN+TR alt kume profilini karsilastirabilir, retrieval method degistirerek sistem davranisini gozlemleyin.</p>
      <form id="search-form">
        <div>
          <label for="query">Query</label>
          <input id="query" name="query" autocomplete="off" placeholder="Orn. I need a dataset with historical earthquake events and seismic activity records." required>
        </div>
        <div>
          <label for="profile">Profile</label>
          <select id="profile" name="profile"></select>
        </div>
        <div>
          <label for="method">Method</label>
          <select id="method" name="method">
            <option value="semantic">Semantic</option>
            <option value="bm25">BM25</option>
            <option value="hybrid" selected>Hybrid</option>
          </select>
        </div>
        <div>
          <label for="source">Source</label>
          <select id="source" name="source">
            <option value="all">All</option>
            <option value="kaggle">Kaggle</option>
            <option value="huggingface">Hugging Face</option>
          </select>
        </div>
        <div>
          <label for="top-k">Top K</label>
          <select id="top-k" name="top_k">
            <option value="5">Top 5</option>
            <option value="10">Top 10</option>
            <option value="15">Top 15</option>
          </select>
        </div>
        <div>
          <button id="submit" type="submit">Search</button>
        </div>
      </form>
      <div class="examples" id="examples" aria-label="Sample queries">
        <button class="example-btn" type="button" data-query="I am looking for a dataset that contains historical earthquake events and seismic activity records from around the world.">earthquake</button>
        <button class="example-btn" type="button" data-query="I need a dataset for detecting audio deepfakes in speech recordings and spoofed voice samples.">audio deepfake</button>
        <button class="example-btn" type="button" data-query="I want a dataset with historical stock prices and daily market data for companies over time.">stock market</button>
        <button class="example-btn" type="button" data-query="I am searching for movie review datasets for sentiment analysis and opinion classification.">movie review</button>
      </div>
      <p class="score-note" id="profile-note">Secilen profile gore semantic temsil degisir. Method secimi ise lexical ve semantic katkilarin davranisini degistirir.</p>
    </section>

    <section class="panel" aria-label="RQ1 Focus">
      <h2>RQ1 Focus</h2>
      <p class="panel-note">Soru 1: Veri seti aciklamalari kullanilarak semantic data discovery gerceklestirilebilir mi ve bu yaklasim lexical baseline ile karsilastirildiginda ne kadar etkili olur? Asagidaki kartlar tezde uretilen RQ1 benchmark sonucunu ozetler.</p>
      <div class="rq1-grid" id="rq1-overview"></div>
      <div class="rq1-grid" id="rq1-style-grid"></div>
      <div class="rq1-banner" id="rq1-banner"></div>
    </section>

    <section class="panel" id="rq1-live-panel" aria-label="RQ1 Live Comparison" hidden>
      <h2>RQ1 Live Comparison</h2>
      <p class="panel-note" id="rq1-live-note">Ayni sorgunun uc retrieval method ile nasil davrandigi burada gorunur.</p>
      <div class="rq1-banner" id="rq1-guidance"></div>
      <div class="rq1-grid" id="rq1-live-grid"></div>
    </section>

    <p id="status" class="status">Ready.</p>
    <section id="results" class="results" aria-live="polite">
      <div class="empty">Bir sorgu girildiginde sonuclar burada listelenir.</div>
    </section>
  </main>

  <div class="modal-backdrop" id="detail-modal" aria-hidden="true">
    <div class="modal" role="dialog" aria-modal="true" aria-labelledby="modal-title">
      <div class="modal-head">
        <h2 id="modal-title">Detail</h2>
        <button class="close-btn" id="modal-close" type="button">Close</button>
      </div>
      <dl id="modal-meta"></dl>
      <section class="modal-section">
        <h3>Why this dataset?</h3>
        <div class="why-box" id="modal-why"></div>
      </section>
      <section class="modal-section">
        <h3>Full description</h3>
        <div class="modal-text" id="modal-text"></div>
      </section>
    </div>
  </div>

  <script>
    const form = document.querySelector("#search-form");
    const queryInput = document.querySelector("#query");
    const profileInput = document.querySelector("#profile");
    const methodInput = document.querySelector("#method");
    const sourceInput = document.querySelector("#source");
    const topKInput = document.querySelector("#top-k");
    const submitButton = document.querySelector("#submit");
    const statusEl = document.querySelector("#status");
    const resultsEl = document.querySelector("#results");
    const statsEl = document.querySelector("#stats");
    const examplesEl = document.querySelector("#examples");
    const profileNoteEl = document.querySelector("#profile-note");
    const rq1OverviewEl = document.querySelector("#rq1-overview");
    const rq1StyleEl = document.querySelector("#rq1-style-grid");
    const rq1BannerEl = document.querySelector("#rq1-banner");
    const rq1LivePanelEl = document.querySelector("#rq1-live-panel");
    const rq1GuidanceEl = document.querySelector("#rq1-guidance");
    const rq1LiveGridEl = document.querySelector("#rq1-live-grid");
    const rq1LiveNoteEl = document.querySelector("#rq1-live-note");
    const detailModal = document.querySelector("#detail-modal");
    const modalTitle = document.querySelector("#modal-title");
    const modalMeta = document.querySelector("#modal-meta");
    const modalWhy = document.querySelector("#modal-why");
    const modalText = document.querySelector("#modal-text");
    const modalClose = document.querySelector("#modal-close");

    const methodLabels = {
      semantic: "Semantic",
      bm25: "BM25",
      hybrid: "Hybrid"
    };

    const sourceLabels = {
      all: "All sources",
      kaggle: "Kaggle",
      huggingface: "Hugging Face"
    };

    let currentResults = [];
    let currentMethod = "hybrid";
    let metadataPayload = null;

    const escapeHtml = (value) => String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");

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

    function scoreLabel(method) {
      if (method === "bm25") return "BM25";
      if (method === "hybrid") return "HYB";
      return "SEM";
    }

    function methodRole(method) {
      const roles = {
        semantic: "Aciklama tabanli semantic retrieval",
        bm25: "Lexical baseline",
        hybrid: "Semantic + lexical birlikte"
      };
      return roles[method] || "";
    }

    function methodBadge(method) {
      return method === "hybrid" ? "En guclu pratik sistem" : method === "semantic" ? "Semantic signal" : "Lexical signal";
    }

    function getProfileMeta(profileKey) {
      return metadataPayload?.profiles?.find((profile) => profile.key === profileKey) || null;
    }

    function renderStats(profileKey) {
      if (!metadataPayload) return;
      const stats = metadataPayload.profile_stats?.[profileKey] || metadataPayload.profile_stats?.[metadataPayload.default_profile] || {};
      const sources = stats.source_counts || {};
      const items = [
        ["Indexed", stats.indexed_rows ?? "-"],
        ["Kaggle", sources.kaggle ?? "-"],
        ["Hugging Face", sources.huggingface ?? "-"]
      ];
      statsEl.innerHTML = items.map(([label, value]) => `
        <div class="stat"><strong>${escapeHtml(value)}</strong><span>${escapeHtml(label)}</span></div>
      `).join("");

      const profile = getProfileMeta(profileKey);
      if (profile) {
        profileNoteEl.textContent = `${profile.label}: ${profile.description}`;
      }
    }

    function findRQ1Row(rows, method) {
      return (rows || []).find((row) => row.method === method) || null;
    }

    function rq1MetricCard(label, value) {
      return `
        <div class="rq1-metric">
          <strong>${escapeHtml(value)}</strong>
          <span>${escapeHtml(label)}</span>
        </div>
      `;
    }

    function renderRQ1Overview() {
      const rq1 = metadataPayload?.rq1;
      if (!rq1) return;

      const methods = ["bm25", "semantic", "hybrid"];
      const overallCards = methods.map((method) => {
        const row = findRQ1Row(rq1.overall, method);
        if (!row) return "";
        return `
          <article class="rq1-card">
            <span class="rq1-label">${escapeHtml(methodBadge(method))}</span>
            <h3>${escapeHtml(methodLabels[method])}</h3>
            <p>${escapeHtml(methodRole(method))}</p>
            <div class="rq1-metrics">
              ${rq1MetricCard("nDCG@5", Number(row.mean_ndcg_at_k).toFixed(3))}
              ${rq1MetricCard("MRR", Number(row.mean_mrr).toFixed(3))}
              ${rq1MetricCard("P@1", Number(row.mean_precision_at_1).toFixed(3))}
            </div>
          </article>
        `;
      }).join("");
      rq1OverviewEl.innerHTML = overallCards;

      const keywordRows = rq1.by_query_style?.keyword || [];
      const sentenceRows = rq1.by_query_style?.sentence || [];
      const keywordHybrid = findRQ1Row(keywordRows, "hybrid");
      const keywordBm25 = findRQ1Row(keywordRows, "bm25");
      const sentenceHybrid = findRQ1Row(sentenceRows, "hybrid");
      const sentenceSemantic = findRQ1Row(sentenceRows, "semantic");
      const sentenceBm25 = findRQ1Row(sentenceRows, "bm25");

      rq1StyleEl.innerHTML = `
        <article class="rq1-card">
          <span class="rq1-label">Keyword Queries</span>
          <h4>Kisa ve anahtar kelime odakli aramalar</h4>
          <p>BM25 burada hala anlamli bir baseline. Ama genel ranking kalitesinde hybrid one geciyor.</p>
          <div class="rq1-metrics">
            ${keywordBm25 ? rq1MetricCard("BM25 MRR", Number(keywordBm25.mean_mrr).toFixed(3)) : ""}
            ${keywordHybrid ? rq1MetricCard("Hybrid nDCG@5", Number(keywordHybrid.mean_ndcg_at_k).toFixed(3)) : ""}
          </div>
        </article>
        <article class="rq1-card">
          <span class="rq1-label">Sentence Queries</span>
          <h4>Dogal dil ve ihtiyac anlatan cümleler</h4>
          <p>RQ1'in asil kazanci burada. Semantic ve ozellikle hybrid, BM25'ten daha guclu davraniyor.</p>
          <div class="rq1-metrics">
            ${sentenceSemantic ? rq1MetricCard("Semantic nDCG@5", Number(sentenceSemantic.mean_ndcg_at_k).toFixed(3)) : ""}
            ${sentenceBm25 ? rq1MetricCard("BM25 nDCG@5", Number(sentenceBm25.mean_ndcg_at_k).toFixed(3)) : ""}
            ${sentenceHybrid ? rq1MetricCard("Hybrid nDCG@5", Number(sentenceHybrid.mean_ndcg_at_k).toFixed(3)) : ""}
          </div>
        </article>
      `;

      rq1BannerEl.innerHTML = `
        <strong>RQ1 ozet sonucu</strong>
        <p>Tezdeki ana Ingilizce benchmarkta <b>hybrid</b> en guclu sistem oldu. Ancak sentence-style sorgularda salt <b>semantic retrieval</b> bile BM25'i geciyor. Bu da veri seti aciklamalariyla semantic discovery yapilabildigini gosteriyor.</p>
      `;
    }

    function populateProfileOptions() {
      const profiles = metadataPayload?.profiles || [];
      profileInput.innerHTML = profiles.map((profile) => `
        <option value="${escapeHtml(profile.key)}" ${profile.key === metadataPayload.default_profile ? "selected" : ""}>
          ${escapeHtml(profile.label)}
        </option>
      `).join("");
    }

    function similarityExplanation(method, score) {
      const formattedScore = formatScore(score);
      if (method === "hybrid") {
        return `Bu sonuc semantic ve BM25 skorlarinin normalize edilip birlestirilmesiyle geldi. Hybrid skor ${formattedScore}.`;
      }
      if (method === "bm25") {
        return `Bu sonuc baslik, aciklama ve anahtar kelime eslesmeleri guclu oldugu icin listelendi. BM25 skoru ${formattedScore}.`;
      }
      return `Bu sonuc sorgu ile dataset aciklamasi anlamsal olarak yakin oldugu icin listelendi. Semantic skor ${formattedScore}.`;
    }

    function renderResults(results, method) {
      currentResults = results;
      currentMethod = method;

      if (!results.length) {
        resultsEl.innerHTML = '<div class="empty">Bu sorgu icin sonuc bulunamadi.</div>';
        return;
      }

      resultsEl.innerHTML = results.map((result, index) => {
        const text = String(result.text || "");
        const summary = text.length > 360 ? `${text.slice(0, 360)}...` : text;
        const url = result.url ? `<a href="${escapeHtml(result.url)}" target="_blank" rel="noreferrer">Open source page</a>` : "URL yok";
        return `
          <article class="result">
            <div class="result-head">
              <h2>${index + 1}. ${escapeHtml(result.title || "Isimsiz veri seti")}</h2>
              <span class="score">${scoreLabel(method)} ${formatScore(result.score)}</span>
            </div>
            <div class="meta">
              <span class="chip">${escapeHtml((result.source || "unknown").toUpperCase())}</span>
              <span class="chip">${escapeHtml(result.ref || "ref yok")}</span>
              <span class="chip">${escapeHtml(result.profile_label || "")}</span>
            </div>
            <p class="summary">${escapeHtml(summary)}</p>
            <div class="actions">
              <button class="detail-btn" type="button" data-result-index="${index}">Detail</button>
              ${url}
            </div>
          </article>
        `;
      }).join("");
    }

    function renderRQ1Live(payload) {
      const rq1 = payload.rq1;
      if (!rq1) {
        rq1LivePanelEl.hidden = true;
        return;
      }

      rq1LivePanelEl.hidden = false;
      rq1LiveNoteEl.textContent = `Bu panel, ayni sorgunun BM25, Semantic ve Hybrid ile nasil davrandigini canli olarak gosterir. Algilanan sorgu tipi: ${rq1.query_style_label}.`;
      rq1GuidanceEl.innerHTML = `
        <strong>Yontem onerisi</strong>
        <p>${escapeHtml(rq1.guidance_text || "")}</p>
      `;

      const cards = (rq1.snapshots || []).map((snapshot) => {
        const top = snapshot.top_result;
        const benchmark = snapshot.benchmark_metrics;
        return `
          <article class="rq1-card">
            <span class="rq1-label">${escapeHtml(methodBadge(snapshot.method))}</span>
            <h3>${escapeHtml(methodLabels[snapshot.method] || snapshot.method)}</h3>
            <p>${escapeHtml(methodRole(snapshot.method))}</p>
            <div class="rq1-metrics">
              ${benchmark ? rq1MetricCard("Benchmark nDCG@5", Number(benchmark.mean_ndcg_at_k).toFixed(3)) : ""}
              ${benchmark ? rq1MetricCard("Benchmark P@1", Number(benchmark.mean_precision_at_1).toFixed(3)) : ""}
              ${top ? rq1MetricCard("Live top score", formatScore(top.score)) : ""}
            </div>
            <p><strong>Top-1:</strong> ${escapeHtml(top?.title || "Sonuc yok")}</p>
            <p>${escapeHtml(top?.source ? top.source.toUpperCase() : "")}${top?.ref ? ` / ${top.ref}` : ""}</p>
          </article>
        `;
      }).join("");
      rq1LiveGridEl.innerHTML = cards;
    }

    function openDetail(index) {
      const result = currentResults[index];
      if (!result) return;

      modalTitle.textContent = result.title || "Isimsiz veri seti";
      const urlValue = result.url
        ? `<a href="${escapeHtml(result.url)}" target="_blank" rel="noreferrer">${escapeHtml(result.url)}</a>`
        : "URL yok";
      const metaItems = [
        ["Baslik", result.title || "Isimsiz veri seti"],
        ["Profile", result.profile_label || "-"],
        ["Method", methodLabels[currentMethod] || currentMethod],
        ["Skor", formatScore(result.score)],
        ["Kaynak", (result.source || "unknown").toUpperCase()],
        ["Referans", result.ref || "ref yok"],
        ["URL", urlValue],
        ["Keywords", formatList(result.keywords)],
        ["License", result.license || "-"],
        ["Description length", `${result.description_len_chars ?? "-"} chars / ${result.description_len_words ?? "-"} words`],
        ["Quality flag", formatList(result.quality_flags, "flag yok")]
      ];
      if (currentMethod === "hybrid") {
        metaItems.push(["Semantic component", formatScore(result.semantic_score)]);
        metaItems.push(["BM25 component", formatScore(result.bm25_score)]);
        metaItems.push(["Weights", `Semantic ${formatScore(result.semantic_weight)} / BM25 ${formatScore(result.bm25_weight)}`]);
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

    async function loadMetadata() {
      const response = await fetch("/api/metadata");
      if (!response.ok) throw new Error("Metadata yuklenemedi.");
      metadataPayload = await response.json();
      populateProfileOptions();
      renderStats(metadataPayload.default_profile);
      renderRQ1Overview();
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

    profileInput.addEventListener("change", () => {
      renderStats(profileInput.value);
    });

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const query = queryInput.value.trim();
      const profile = profileInput.value;
      const method = methodInput.value;
      const source = sourceInput.value;
      const topK = Number(topKInput.value);
      if (!query) return;

      submitButton.disabled = true;
      statusEl.className = "status";
      statusEl.textContent = "Searching...";

      try {
        const response = await fetch("/api/search", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query, profile, method, source, top_k: topK })
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || "Arama tamamlanamadi.");

        renderResults(payload.results || [], payload.method);
        renderRQ1Live(payload);
        renderStats(payload.profile.key);
        const sourceLabel = sourceLabels[payload.source] || "All sources";
        const conciseIntent = payload.query_plan?.concise_intent;
        const intentNote = conciseIntent ? ` Intent: ${conciseIntent}.` : "";
        statusEl.textContent = `${payload.profile.label} / ${methodLabels[payload.method] || payload.method} / ${sourceLabel} ile "${payload.query}" icin ${payload.results.length} sonuc listelendi.${intentNote}`;
      } catch (error) {
        statusEl.className = "status error";
        statusEl.textContent = error.message;
      } finally {
        submitButton.disabled = false;
      }
    });

    loadMetadata()
      .then(() => {
        queryInput.focus();
      })
      .catch((error) => {
        statusEl.className = "status error";
        statusEl.textContent = error.message;
      });
  </script>
</body>
</html>
"""


def configure_output():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def load_json_file(path: Path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_app_metadata():
    profile_entries = []
    profile_stats = {}

    for profile in list_profiles():
        metadata = load_json_file(get_profile_paths(profile.key)["index_metadata"], {})
        profile_stats[profile.key] = metadata
        profile_entries.append(
            {
                "key": profile.key,
                "label": profile.label,
                "model_name": profile.model_name,
                "description": profile.description,
                "indexed_rows": metadata.get("indexed_rows"),
                "source_counts": metadata.get("source_counts") or {},
            }
        )

    default_stats = profile_stats.get(DEFAULT_WEB_PROFILE, {})
    return {
        "default_profile": DEFAULT_WEB_PROFILE,
        "profiles": profile_entries,
        "profile_stats": profile_stats,
        "indexed_rows": default_stats.get("indexed_rows"),
        "source_counts": default_stats.get("source_counts") or {},
        "rq1": load_rq1_metadata(),
    }


def load_rq1_metadata():
    overall = load_json_file(RQ1_SUMMARY_PATH, [])
    by_query_style_rows = load_json_file(RQ1_QUERY_STYLE_PATH, [])
    by_benchmark_rows = load_json_file(RQ1_BENCHMARK_PATH, [])
    by_query_style = {}
    for row in by_query_style_rows:
        key = row.get("query_style")
        if not key:
            continue
        by_query_style.setdefault(key, []).append(row)
    return {
        "overall": overall,
        "by_query_style": by_query_style,
        "by_benchmark": by_benchmark_rows,
    }


class AppState:
    search_engines = {}
    bm25_indices = {}
    metadata = {}
    lock = threading.Lock()


def ensure_search_engine(profile_key):
    profile = get_profile(profile_key)
    cached = AppState.search_engines.get(profile.key)
    if cached is not None:
        return cached

    with AppState.lock:
        cached = AppState.search_engines.get(profile.key)
        if cached is not None:
            return cached
        print(f"[WEB] Search engine yukleniyor: {profile.key}")
        AppState.search_engines[profile.key] = load_search_engine(profile.key)
        return AppState.search_engines[profile.key]


def ensure_bm25_engine(profile_key):
    profile = get_profile(profile_key)
    cached = AppState.bm25_indices.get(profile.key)
    if cached is not None:
        return cached

    with AppState.lock:
        cached = AppState.bm25_indices.get(profile.key)
        if cached is not None:
            return cached

        engine = AppState.search_engines.get(profile.key)
        if engine is not None:
            _, _, mappings = engine
        else:
            mappings = load_mappings(profile.key)
        print(f"[WEB] BM25 index hazirlaniyor: {profile.key}")
        AppState.bm25_indices[profile.key] = BM25Index(mappings)
        return AppState.bm25_indices[profile.key]


def filter_results_by_source(raw_results, source_filter, top_k):
    results = []
    for score, item in raw_results:
        if source_filter != "all" and (item.get("source") or "").lower() != source_filter:
            continue
        results.append((score, item))
        if len(results) >= top_k:
            break
    return results


def execute_search_method(method, query, profile_key, source_filter, top_k, query_plan):
    if method == "bm25":
        bm25_index = ensure_bm25_engine(profile_key)
        candidate_k = len(bm25_index.doc_ids) if source_filter != "all" else top_k
        raw_results = bm25_search(
            query,
            bm25_index,
            top_k=candidate_k,
            query_plan=query_plan,
        )
    elif method == "hybrid":
        model, index, mappings = ensure_search_engine(profile_key)
        bm25_index = ensure_bm25_engine(profile_key)
        candidate_k = (
            len(mappings)
            if source_filter != "all"
            else max(top_k, min(DEFAULT_CANDIDATE_K, len(mappings)))
        )
        raw_results = hybrid_search(
            query,
            model,
            index,
            mappings,
            bm25_index,
            top_k=candidate_k if source_filter != "all" else top_k,
            semantic_weight=DEFAULT_SEMANTIC_WEIGHT,
            candidate_k=candidate_k,
            profile_key=profile_key,
            query_plan=query_plan,
        )
    else:
        model, index, mappings = ensure_search_engine(profile_key)
        candidate_k = index.ntotal if source_filter != "all" else top_k
        raw_results = semantic_search(
            query,
            model,
            index,
            mappings,
            top_k=candidate_k,
            profile_key=profile_key,
            query_plan=query_plan,
        )
    return filter_results_by_source(raw_results, source_filter, top_k)


def find_method_row(rows, method):
    for row in rows:
        if row.get("method") == method:
            return row
    return None


def build_rq1_payload(query_plan, profile_key, source_filter, top_k, method_results):
    rq1_data = (AppState.metadata or {}).get("rq1") or {}
    query_style = "sentence" if query_plan.get("is_sentence_query") else "keyword"
    style_rows = (rq1_data.get("by_query_style") or {}).get(query_style, [])
    recommended = max(
        style_rows,
        key=lambda row: row.get("mean_ndcg_at_k", 0.0),
        default=None,
    )
    guidance = []
    if recommended:
        guidance.append(
            f"Bu sorgu `{query_style}` tipinde gorunuyor. RQ1 benchmarkinda bu tipte en guclu method `{METHOD_LABELS.get(recommended['method'], recommended['method'])}` "
            f"(nDCG@5={recommended['mean_ndcg_at_k']:.3f})."
        )

    semantic_row = find_method_row(style_rows, "semantic")
    bm25_row = find_method_row(style_rows, "bm25")
    if semantic_row and bm25_row and query_style == "sentence":
        guidance.append(
            f"Sentence query'lerde semantic retrieval, BM25'i geciyor "
            f"(semantic nDCG@5={semantic_row['mean_ndcg_at_k']:.3f}, BM25={bm25_row['mean_ndcg_at_k']:.3f})."
        )
    elif semantic_row and bm25_row:
        guidance.append(
            f"Keyword query'lerde BM25 hala anlamli bir baseline; ama genel ranking kalitesinde hybrid onde kalir."
        )

    if profile_key != DEFAULT_WEB_PROFILE:
        guidance.append(
            "Not: RQ1 benchmark metrikleri tezin ana Ingilizce MiniLM profiline aittir."
        )

    snapshots = []
    for method in ("bm25", "semantic", "hybrid"):
        rows = method_results.get(method) or []
        top = rows[0] if rows else None
        top_result = None
        if top is not None:
            score, item = top
            top_result = {
                "score": score,
                "title": item.get("title"),
                "source": item.get("source"),
                "ref": item.get("ref"),
            }
        snapshots.append(
            {
                "method": method,
                "top_result": top_result,
                "benchmark_metrics": find_method_row(style_rows, method),
            }
        )

    return {
        "query_style": query_style,
        "query_style_label": "sentence query" if query_style == "sentence" else "keyword query",
        "recommended_method": recommended.get("method") if recommended else None,
        "guidance_text": " ".join(guidance),
        "source_filter": source_filter,
        "top_k": top_k,
        "snapshots": snapshots,
    }


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "SemanticDatasetSearch/3.0"

    def log_message(self, fmt, *args):
        print(f"[WEB] {self.address_string()} - {fmt % args}")

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
            self.send_json(AppState.metadata)
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
            self.send_json({"error": "Sorgu bos olamaz."}, HTTPStatus.BAD_REQUEST)
            return

        try:
            profile = get_profile(payload.get("profile", DEFAULT_WEB_PROFILE))
        except ValueError as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
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
            query_plan = build_query_plan(query)
            method_results = {
                candidate_method: execute_search_method(
                    candidate_method,
                    query,
                    profile.key,
                    source_filter,
                    top_k,
                    query_plan,
                )
                for candidate_method in ("bm25", "semantic", "hybrid")
            }
        except Exception as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        filtered_results = method_results[method]
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
                    "semantic_variant_hits": item.get("semantic_variant_hits"),
                    "profile_key": profile.key,
                    "profile_label": profile.label,
                }
            )

        self.send_json(
            {
                "query": query,
                "query_plan": query_plan,
                "method": method,
                "profile": {"key": profile.key, "label": profile.label},
                "source": source_filter,
                "top_k": top_k,
                "rq1": build_rq1_payload(query_plan, profile.key, source_filter, top_k, method_results),
                "results": results,
            }
        )


def main():
    configure_output()
    AppState.metadata = load_app_metadata()

    server = ThreadingHTTPServer((HOST, PORT), RequestHandler)
    print(f"[WEB] UI hazir: http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\\n[WEB] Kapatiliyor...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
