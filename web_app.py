import json
import re
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
RQ2_REPORT_DIR = BASE_DIR / "reports" / "evaluation" / "rq2_cross_source"
RQ2_SUMMARY_PATH = RQ2_REPORT_DIR / "evaluation_summary.json"
RQ2_DIRECTION_PATH = RQ2_REPORT_DIR / "evaluation_by_direction.json"
RQ2_TOPIC_PATH = RQ2_REPORT_DIR / "evaluation_by_topic.json"
RQ3_REPORT_DIR = BASE_DIR / "reports" / "evaluation" / "rq3_model_effect"
RQ3_SUMMARY_PATH = RQ3_REPORT_DIR / "evaluation_summary.json"
RQ3_LANGUAGE_PATH = RQ3_REPORT_DIR / "evaluation_by_language.json"
RQ3_BENCHMARK_PATH = RQ3_REPORT_DIR / "evaluation_by_benchmark.json"
RQ3_STUDY_SLICE_PATH = RQ3_REPORT_DIR / "evaluation_by_study_slice.json"
RQ4_REPORT_DIR = BASE_DIR / "reports" / "evaluation" / "rq4_description_quality"
RQ4_DETAILS_PATH = RQ4_REPORT_DIR / "evaluation_details.json"
RQ4_STYLE_PATH = RQ4_REPORT_DIR / "rq4_by_description_style.json"
RQ4_LENGTH_PATH = RQ4_REPORT_DIR / "rq4_by_length_bucket.json"
RQ4_TERM_PATH = RQ4_REPORT_DIR / "rq4_by_term_bucket.json"
RQ4_CORPUS_DISTRIBUTION_PATH = RQ4_REPORT_DIR / "rq4_corpus_feature_distribution.json"
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
DIRECTION_LABELS = {
    "huggingface_to_kaggle": "Hugging Face -> Kaggle",
    "kaggle_to_huggingface": "Kaggle -> Hugging Face",
}
TOPIC_LABELS = {
    "amazon_reviews": "Amazon reviews",
    "climate_change": "Climate change",
    "movie_reviews": "Movie reviews",
    "quran_text_audio": "Quran text and audio",
    "sentiment_analysis": "Sentiment analysis",
    "turkish_product_reviews": "Turkish product reviews",
}
STUDY_SLICE_LABELS = {
    "english_main": "English main",
    "cross_source": "Cross-source",
    "tr_subset": "Turkish subset",
}
LANGUAGE_LABELS = {
    "en": "English",
    "tr": "Turkish",
}
PROFILE_ORDER = ("minilm", "multilingual")
DESCRIPTION_STYLE_LABELS = {
    "metadata_heavy": "Metadata heavy",
    "mixed_structured": "Mixed structured",
    "narrative": "Narrative",
}
LENGTH_BUCKET_LABELS = {
    "short": "Short",
    "medium": "Medium",
    "long": "Long",
}
TERM_BUCKET_LABELS = {
    "term_sparse": "Term sparse",
    "term_moderate": "Term moderate",
    "term_rich": "Term rich",
}
CONTENT_TERM_RE = re.compile(r"[0-9a-zA-ZçğıöşüÇĞİÖŞÜ]{3,}", re.IGNORECASE)
QUALITY_FLAG_META = {
    "empty": {
        "label": "Empty description",
        "severity": "risk",
        "message": "Kayit aciklama tasimiyor; semantic sinyal cok zayif.",
    },
    "no_long_description": {
        "label": "No long description",
        "severity": "warning",
        "message": "Ayrintili aciklama eksik; kayit daha cok subtitle veya keyword uzerinden temsil edildi.",
    },
    "keyword_only": {
        "label": "Keyword only",
        "severity": "risk",
        "message": "Kayit agirlikla anahtar kelime veya etiket seviyesinde; dogal dil aciklamasi zayif.",
    },
    "short_description": {
        "label": "Short description",
        "severity": "warning",
        "message": "Aciklama kisa; semantic retrieval icin tasidigi baglam sinirli.",
    },
    "too_short": {
        "label": "Too short",
        "severity": "warning",
        "message": "Temizleme asamasinda metin cok kisa gorunmus.",
    },
    "short_context": {
        "label": "Short context",
        "severity": "warning",
        "message": "Canli tahmine gore aciklama kisa baglam tasiyor.",
    },
    "metadata_heavy": {
        "label": "Metadata heavy",
        "severity": "warning",
        "message": "Aciklama alan listesi veya metadata agirlikli; anlamsal akis zayif olabilir.",
    },
    "term_sparse": {
        "label": "Term sparse",
        "severity": "warning",
        "message": "Icerik terimi sayisi dusuk; semantic ayirt edicilik azalabilir.",
    },
    "low_information": {
        "label": "Low information",
        "severity": "risk",
        "message": "Kisa, metadata agirlikli veya keyword benzeri kayit; retrieval kalitesi icin riskli.",
    },
}
CONTENT_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "dataset",
    "data",
    "are",
    "into",
    "over",
    "under",
    "between",
    "about",
    "your",
    "their",
    "ve",
    "bir",
    "icin",
    "için",
    "olan",
    "veri",
    "seti",
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

    .tabbar {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 0 0 18px;
    }

    .tab-btn {
      width: auto;
      min-height: 40px;
      margin: 0;
      padding: 9px 14px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #fff;
      color: var(--accent-strong);
      font-size: 14px;
      font-weight: 700;
    }

    .tab-btn:hover {
      border-color: var(--accent);
      background: var(--soft);
    }

    .tab-btn[aria-selected="true"] {
      border-color: var(--accent-strong);
      background: var(--accent);
      color: #fff;
    }

    .tab-panel[hidden] {
      display: none;
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

    .chip-warning {
      background: #fce8c8;
      color: #6a4308;
    }

    .chip-risk {
      background: #f7d9dd;
      color: #7a1730;
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

    .flag-list {
      display: grid;
      gap: 8px;
    }

    .flag-item {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfb;
      padding: 10px;
    }

    .flag-item strong {
      display: block;
      margin-bottom: 4px;
    }

    .flag-item p {
      margin: 0;
      color: #2c332f;
      line-height: 1.45;
      font-size: 14px;
    }

    .flag-meta {
      display: inline-block;
      margin-top: 6px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.02em;
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

    <div class="tabbar" role="tablist" aria-label="Main views">
      <button class="tab-btn" type="button" role="tab" aria-selected="true" aria-controls="tab-search" id="tab-btn-search" data-tab="search">Search</button>
      <button class="tab-btn" type="button" role="tab" aria-selected="false" aria-controls="tab-rq1" id="tab-btn-rq1" data-tab="rq1">RQ1</button>
      <button class="tab-btn" type="button" role="tab" aria-selected="false" aria-controls="tab-rq2" id="tab-btn-rq2" data-tab="rq2">RQ2</button>
      <button class="tab-btn" type="button" role="tab" aria-selected="false" aria-controls="tab-rq3" id="tab-btn-rq3" data-tab="rq3">RQ3</button>
      <button class="tab-btn" type="button" role="tab" aria-selected="false" aria-controls="tab-rq4" id="tab-btn-rq4" data-tab="rq4">RQ4</button>
    </div>

    <section class="tab-panel" id="tab-search" role="tabpanel" aria-labelledby="tab-btn-search">
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

      <p id="status" class="status">Ready.</p>
      <section id="results" class="results" aria-live="polite">
        <div class="empty">Bir sorgu girildiginde sonuclar burada listelenir.</div>
      </section>
    </section>

    <section class="tab-panel" id="tab-rq1" role="tabpanel" aria-labelledby="tab-btn-rq1" hidden>
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
    </section>

    <section class="tab-panel" id="tab-rq2" role="tabpanel" aria-labelledby="tab-btn-rq2" hidden>
      <section class="panel" aria-label="RQ2 Cross-Source Focus">
        <h2>RQ2 Cross-Source Focus</h2>
        <p class="panel-note">Soru 2: Bir platformdaki veri seti ihtiyaci, diger platformdaki anlamsal olarak benzer veri setlerine ne kadar iyi koprulenebiliyor? Bu panel, cross-source retrieval benchmarkinin genel sonucunu, yon farkini ve konu bazli kirilimlarini ozetler.</p>
        <div class="rq1-grid" id="rq2-composition-grid"></div>
        <div class="rq1-grid" id="rq2-overview"></div>
        <div class="rq1-grid" id="rq2-direction-grid"></div>
        <div class="rq1-grid" id="rq2-topic-grid"></div>
        <div class="rq1-banner" id="rq2-banner"></div>
      </section>

      <section class="panel" id="rq2-live-panel" aria-label="RQ2 Live Comparison" hidden>
        <h2>RQ2 Live Comparison</h2>
        <p class="panel-note" id="rq2-live-note">Kaynak filtresi tek platforma indirildiginde, bu panel cross-source benchmarka en yakin canli yorumu gosterir.</p>
        <div class="rq1-banner" id="rq2-guidance"></div>
        <div class="rq1-grid" id="rq2-live-grid"></div>
      </section>
    </section>

    <section class="tab-panel" id="tab-rq3" role="tabpanel" aria-labelledby="tab-btn-rq3" hidden>
      <section class="panel" aria-label="RQ3 Model Effect Focus">
        <h2>RQ3 Model Effect Focus</h2>
        <p class="panel-note">Soru 3: Dil modeli tipi semantic similarity sonucunu nasil etkiliyor? Bu panel, MiniLM ve multilingual profile'i ayni benchmark ailesi uzerinde karsilastirir.</p>
        <div class="rq1-grid" id="rq3-composition-grid"></div>
        <div class="rq1-grid" id="rq3-overview-grid"></div>
        <div class="rq1-grid" id="rq3-slice-grid"></div>
        <div class="rq1-grid" id="rq3-language-grid"></div>
        <div class="rq1-banner" id="rq3-banner"></div>
      </section>

      <section class="panel" id="rq3-live-panel" aria-label="RQ3 Live Comparison" hidden>
        <h2>RQ3 Live Comparison</h2>
        <p class="panel-note" id="rq3-live-note">Secilen profile ve sorgu diline gore model etkisini canli olarak yorumlar.</p>
        <div class="rq1-banner" id="rq3-guidance"></div>
        <div class="rq1-grid" id="rq3-live-grid"></div>
      </section>
    </section>

    <section class="tab-panel" id="tab-rq4" role="tabpanel" aria-labelledby="tab-btn-rq4" hidden>
      <section class="panel" aria-label="RQ4 Description Quality Focus">
        <h2>RQ4 Description Quality Focus</h2>
        <p class="panel-note">Soru 4: Aciklama dili, uzunlugu ve icerilen terimler semantic temsil kalitesini nasil etkiliyor? Bu panel, belge kalitesinin retrieval basarisina etkisini ozetler.</p>
        <div class="rq1-grid" id="rq4-composition-grid"></div>
        <div class="rq1-grid" id="rq4-style-grid"></div>
        <div class="rq1-grid" id="rq4-length-grid"></div>
        <div class="rq1-grid" id="rq4-term-grid"></div>
        <div class="rq1-banner" id="rq4-banner"></div>
      </section>

      <section class="panel" id="rq4-live-panel" aria-label="RQ4 Live Comparison" hidden>
        <h2>RQ4 Live Comparison</h2>
        <p class="panel-note" id="rq4-live-note">Canli sonuclardaki ust siradaki veri setlerinin belge kalitesi burada benchmark diline cevrilir.</p>
        <div class="rq1-banner" id="rq4-guidance"></div>
        <div class="rq1-grid" id="rq4-live-grid"></div>
      </section>
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
        <h3>Quality flags</h3>
        <div class="flag-list" id="modal-flags"></div>
      </section>
      <section class="modal-section">
        <h3>Full description</h3>
        <div class="modal-text" id="modal-text"></div>
      </section>
    </div>
  </div>

  <script>
    const tabButtons = Array.from(document.querySelectorAll(".tab-btn"));
    const tabPanels = Array.from(document.querySelectorAll(".tab-panel"));
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
    const rq2CompositionEl = document.querySelector("#rq2-composition-grid");
    const rq2OverviewEl = document.querySelector("#rq2-overview");
    const rq2DirectionEl = document.querySelector("#rq2-direction-grid");
    const rq2TopicEl = document.querySelector("#rq2-topic-grid");
    const rq2BannerEl = document.querySelector("#rq2-banner");
    const rq2LivePanelEl = document.querySelector("#rq2-live-panel");
    const rq2GuidanceEl = document.querySelector("#rq2-guidance");
    const rq2LiveGridEl = document.querySelector("#rq2-live-grid");
    const rq2LiveNoteEl = document.querySelector("#rq2-live-note");
    const rq3CompositionEl = document.querySelector("#rq3-composition-grid");
    const rq3OverviewEl = document.querySelector("#rq3-overview-grid");
    const rq3SliceEl = document.querySelector("#rq3-slice-grid");
    const rq3LanguageEl = document.querySelector("#rq3-language-grid");
    const rq3BannerEl = document.querySelector("#rq3-banner");
    const rq3LivePanelEl = document.querySelector("#rq3-live-panel");
    const rq3GuidanceEl = document.querySelector("#rq3-guidance");
    const rq3LiveGridEl = document.querySelector("#rq3-live-grid");
    const rq3LiveNoteEl = document.querySelector("#rq3-live-note");
    const rq4CompositionEl = document.querySelector("#rq4-composition-grid");
    const rq4StyleEl = document.querySelector("#rq4-style-grid");
    const rq4LengthEl = document.querySelector("#rq4-length-grid");
    const rq4TermEl = document.querySelector("#rq4-term-grid");
    const rq4BannerEl = document.querySelector("#rq4-banner");
    const rq4LivePanelEl = document.querySelector("#rq4-live-panel");
    const rq4GuidanceEl = document.querySelector("#rq4-guidance");
    const rq4LiveGridEl = document.querySelector("#rq4-live-grid");
    const rq4LiveNoteEl = document.querySelector("#rq4-live-note");
    const detailModal = document.querySelector("#detail-modal");
    const modalTitle = document.querySelector("#modal-title");
    const modalMeta = document.querySelector("#modal-meta");
    const modalWhy = document.querySelector("#modal-why");
    const modalFlags = document.querySelector("#modal-flags");
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

    const directionLabels = {
      huggingface_to_kaggle: "Hugging Face -> Kaggle",
      kaggle_to_huggingface: "Kaggle -> Hugging Face"
    };

    const topicLabels = {
      amazon_reviews: "Amazon reviews",
      climate_change: "Climate change",
      movie_reviews: "Movie reviews",
      quran_text_audio: "Quran text and audio",
      sentiment_analysis: "Sentiment analysis",
      turkish_product_reviews: "Turkish product reviews"
    };

    const studySliceLabels = {
      english_main: "English main",
      cross_source: "Cross-source",
      tr_subset: "Turkish subset"
    };

    const languageLabels = {
      en: "English",
      tr: "Turkish"
    };

    const descriptionStyleLabels = {
      metadata_heavy: "Metadata heavy",
      mixed_structured: "Mixed structured",
      narrative: "Narrative"
    };

    const lengthBucketLabels = {
      short: "Short",
      medium: "Medium",
      long: "Long"
    };

    const termBucketLabels = {
      term_sparse: "Term sparse",
      term_moderate: "Term moderate",
      term_rich: "Term rich"
    };

    let currentResults = [];
    let currentMethod = "hybrid";
    let metadataPayload = null;
    let activeTab = "search";

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

    function qualityChipClass(status) {
      if (status === "risk") return "chip chip-risk";
      if (status === "warning") return "chip chip-warning";
      return "chip";
    }

    function renderQualityFlags(details) {
      if (!Array.isArray(details) || !details.length) {
        return '<div class="empty">Bu kayit icin aktif kalite flagi yok.</div>';
      }
      return details.map((detail) => `
        <article class="flag-item">
          <strong>${escapeHtml(detail.label || detail.code || "Flag")}</strong>
          <p>${escapeHtml(detail.message || "")}</p>
          <span class="flag-meta">${escapeHtml(detail.severity || "info")} / ${escapeHtml(detail.origin || "stored")}</span>
        </article>
      `).join("");
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

    function findMethodRow(rows, method) {
      return (rows || []).find((row) => row.method === method) || null;
    }

    function findProfileMethodRow(rows, profile, method) {
      return (rows || []).find((row) => row.profile === profile && row.method === method) || null;
    }

    function profileLabel(profileKey) {
      return getProfileMeta(profileKey)?.label || profileKey || "-";
    }

    function formatDirectionLabel(direction) {
      return directionLabels[direction] || direction || "-";
    }

    function formatTopicLabel(topic) {
      if (topicLabels[topic]) return topicLabels[topic];
      return String(topic || "-")
        .split("_")
        .filter(Boolean)
        .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
        .join(" ");
    }

    function formatStudySliceLabel(slice) {
      return studySliceLabels[slice] || slice || "-";
    }

    function formatLanguageLabel(language) {
      return languageLabels[language] || language || "-";
    }

    function formatDescriptionStyleLabel(style) {
      return descriptionStyleLabels[style] || style || "-";
    }

    function formatLengthBucketLabel(bucket) {
      return lengthBucketLabels[bucket] || bucket || "-";
    }

    function formatTermBucketLabel(bucket) {
      return termBucketLabels[bucket] || bucket || "-";
    }

    function countObjectValues(value) {
      return Object.values(value || {}).reduce((total, item) => total + Number(item || 0), 0);
    }

    function activateTab(tabKey) {
      activeTab = tabKey;
      tabButtons.forEach((button) => {
        const isActive = button.dataset.tab === tabKey;
        button.setAttribute("aria-selected", isActive ? "true" : "false");
      });
      tabPanels.forEach((panel) => {
        panel.hidden = panel.id !== `tab-${tabKey}`;
      });
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
        const row = findMethodRow(rq1.overall, method);
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
      const keywordHybrid = findMethodRow(keywordRows, "hybrid");
      const keywordBm25 = findMethodRow(keywordRows, "bm25");
      const sentenceHybrid = findMethodRow(sentenceRows, "hybrid");
      const sentenceSemantic = findMethodRow(sentenceRows, "semantic");
      const sentenceBm25 = findMethodRow(sentenceRows, "bm25");

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

    function selectTopicInsight(rq2, kind, excludeTopics = []) {
      const entries = Object.entries(rq2?.by_topic || {}).map(([topic, rows]) => {
        const semantic = findMethodRow(rows, "semantic");
        const bm25 = findMethodRow(rows, "bm25");
        const hybrid = findMethodRow(rows, "hybrid");
        let score = Number.NEGATIVE_INFINITY;
        if (kind === "strong") {
          score = semantic?.mean_ndcg_at_k ?? hybrid?.mean_ndcg_at_k ?? score;
        } else if (kind === "hard") {
          score = -(semantic?.mean_ndcg_at_k ?? hybrid?.mean_ndcg_at_k ?? Number.POSITIVE_INFINITY);
        } else if (kind === "gap") {
          score = (semantic?.mean_ndcg_at_k ?? Number.NEGATIVE_INFINITY) - (bm25?.mean_ndcg_at_k ?? 0);
        }
        return { topic, rows, semantic, bm25, hybrid, score };
      });

      return entries
        .filter((entry) => !excludeTopics.includes(entry.topic) && Number.isFinite(entry.score))
        .sort((left, right) => right.score - left.score)[0] || null;
    }

    function renderRQ2Overview() {
      const rq2 = metadataPayload?.rq2;
      if (!rq2) return;

      rq2CompositionEl.innerHTML = `
        <article class="rq1-card">
          <span class="rq1-label">Benchmark Size</span>
          <h4>${escapeHtml(String(rq2.query_count ?? "-"))} anchor query</h4>
          <p>Cross-source benchmark, iki platform arasinda manuel eslestirilmis anchor ciftlerinden olusuyor.</p>
          <div class="rq1-metrics">
            ${rq1MetricCard("Directions", String(Object.keys(rq2.direction_counts || {}).length))}
            ${rq1MetricCard("Topics", String(Object.keys(rq2.topic_counts || {}).length))}
          </div>
        </article>
        <article class="rq1-card">
          <span class="rq1-label">Direction Mix</span>
          <h4>Denge dagilimi</h4>
          <p>Benchmark, her iki yone de bakiyor; boylece kaynak stilinin retrieval uzerindeki etkisi ayrisiyor.</p>
          <div class="rq1-metrics">
            ${rq1MetricCard("HF -> Kaggle", String(rq2.direction_counts?.huggingface_to_kaggle ?? "-"))}
            ${rq1MetricCard("Kaggle -> HF", String(rq2.direction_counts?.kaggle_to_huggingface ?? "-"))}
          </div>
        </article>
        <article class="rq1-card">
          <span class="rq1-label">Topic Coverage</span>
          <h4>${escapeHtml(String(countObjectValues(rq2.topic_counts)))} toplam eslesme</h4>
          <p>Konu cesitliligi, semantic koprunun sadece tek bir problem alanina bagli olup olmadigini test ediyor.</p>
          <div class="rq1-metrics">
            ${rq1MetricCard("Largest slice", formatTopicLabel(rq2.largest_topic || "-"))}
            ${rq1MetricCard("Smallest slice", formatTopicLabel(rq2.smallest_topic || "-"))}
          </div>
        </article>
      `;

      const methods = ["bm25", "semantic", "hybrid"];
      rq2OverviewEl.innerHTML = methods.map((method) => {
        const row = findMethodRow(rq2.overall, method);
        if (!row) return "";
        return `
          <article class="rq1-card">
            <span class="rq1-label">${escapeHtml(methodBadge(method))}</span>
            <h3>${escapeHtml(methodLabels[method])}</h3>
            <p>Cross-source eslesmede ${escapeHtml(methodRole(method))} davranisi.</p>
            <div class="rq1-metrics">
              ${rq1MetricCard("Bridge@5", Number(row.mean_hit_rate_at_k).toFixed(3))}
              ${rq1MetricCard("nDCG@5", Number(row.mean_ndcg_at_k).toFixed(3))}
              ${rq1MetricCard("Bridge@1", Number(row.mean_precision_at_1).toFixed(3))}
            </div>
          </article>
        `;
      }).join("");

      const directions = ["huggingface_to_kaggle", "kaggle_to_huggingface"];
      rq2DirectionEl.innerHTML = directions.map((direction) => {
        const rows = rq2.by_direction?.[direction] || [];
        const semantic = findMethodRow(rows, "semantic");
        const bm25 = findMethodRow(rows, "bm25");
        const hybrid = findMethodRow(rows, "hybrid");
        const best = [...rows].sort((left, right) => (right.mean_ndcg_at_k ?? 0) - (left.mean_ndcg_at_k ?? 0))[0];
        if (!best) return "";
        return `
          <article class="rq1-card">
            <span class="rq1-label">${escapeHtml(formatDirectionLabel(direction))}</span>
            <h4>Yon farki burada belirginlesiyor</h4>
            <p>Bu slice'ta en guclu method <b>${escapeHtml(methodLabels[best.method] || best.method)}</b>. Platformlarin aciklama stili retrieval kalitesini etkiliyor.</p>
            <div class="rq1-metrics">
              ${semantic ? rq1MetricCard("Semantic nDCG@5", Number(semantic.mean_ndcg_at_k).toFixed(3)) : ""}
              ${hybrid ? rq1MetricCard("Hybrid Bridge@5", Number(hybrid.mean_hit_rate_at_k).toFixed(3)) : ""}
              ${bm25 ? rq1MetricCard("BM25 nDCG@5", Number(bm25.mean_ndcg_at_k).toFixed(3)) : ""}
            </div>
          </article>
        `;
      }).join("");

      const strongest = selectTopicInsight(rq2, "strong");
      const hardest = selectTopicInsight(rq2, "hard", strongest ? [strongest.topic] : []);
      const gap = selectTopicInsight(
        rq2,
        "gap",
        [strongest?.topic, hardest?.topic].filter(Boolean)
      );
      const topicInsights = [
        strongest && {
          label: "Strong Bridge",
          title: formatTopicLabel(strongest.topic),
          copy: "Kaynaklar arasi semantik eslesme bu konuda daha kararli. Embedding tabanli retrieval dogrudan kopru kurabiliyor.",
          metrics: [
            strongest.semantic ? ["Semantic nDCG@5", Number(strongest.semantic.mean_ndcg_at_k).toFixed(3)] : null,
            strongest.hybrid ? ["Hybrid Bridge@5", Number(strongest.hybrid.mean_hit_rate_at_k).toFixed(3)] : null
          ].filter(Boolean)
        },
        hardest && {
          label: "Hard Bridge",
          title: formatTopicLabel(hardest.topic),
          copy: "Bu konuda platformlar arasi aciklama farki daha sert. BM25 genelde kopru kuramiyor; semantic ve hybrid ancak kismi toparliyor.",
          metrics: [
            hardest.semantic ? ["Semantic nDCG@5", Number(hardest.semantic.mean_ndcg_at_k).toFixed(3)] : null,
            hardest.bm25 ? ["BM25 nDCG@5", Number(hardest.bm25.mean_ndcg_at_k).toFixed(3)] : null
          ].filter(Boolean)
        },
        gap && {
          label: "Lexical Gap",
          title: formatTopicLabel(gap.topic),
          copy: "Lexical eslesmenin yetersiz kaldigi ama semantic benzerligin fark yarattigi slice. RQ2'nin tez argumani burada netlesiyor.",
          metrics: [
            gap.semantic ? ["Semantic nDCG@5", Number(gap.semantic.mean_ndcg_at_k).toFixed(3)] : null,
            gap.bm25 ? ["BM25 nDCG@5", Number(gap.bm25.mean_ndcg_at_k).toFixed(3)] : null
          ].filter(Boolean)
        }
      ].filter(Boolean);

      rq2TopicEl.innerHTML = topicInsights.map((insight) => `
        <article class="rq1-card">
          <span class="rq1-label">${escapeHtml(insight.label)}</span>
          <h4>${escapeHtml(insight.title)}</h4>
          <p>${escapeHtml(insight.copy)}</p>
          <div class="rq1-metrics">
            ${insight.metrics.map(([label, value]) => rq1MetricCard(label, value)).join("")}
          </div>
        </article>
      `).join("");

      rq2BannerEl.innerHTML = `
        <strong>RQ2 ozet sonucu</strong>
        <p>Cross-source retrieval'da <b>semantic</b> ve <b>hybrid</b>, BM25'e gore acik bicimde daha iyi kopru kuruyor. Tez icin dogrudan cevap semantic retrieval; uygulamadaki en guclu sistem ise hybrid. Kaynak yonu degistikce performans da degisiyor.</p>
      `;
    }

    function renderRQ3Overview() {
      const rq3 = metadataPayload?.rq3;
      if (!rq3) return;

      rq3CompositionEl.innerHTML = `
        <article class="rq1-card">
          <span class="rq1-label">Benchmark Size</span>
          <h4>${escapeHtml(String(rq3.query_count ?? "-"))} total query</h4>
          <p>RQ3, ayni retrieval gorevlerinde farkli semantic profile'larin etkisini olcuyor.</p>
          <div class="rq1-metrics">
            ${rq1MetricCard("Study slices", String(Object.keys(rq3.study_slice_counts || {}).length))}
            ${rq1MetricCard("Languages", String(Object.keys(rq3.language_counts || {}).length))}
          </div>
        </article>
        <article class="rq1-card">
          <span class="rq1-label">Profile Set</span>
          <h4>Iki aktif model profili</h4>
          <p>MiniLM tezdeki ana English baseline; multilingual ise EN+TR genisletme profili olarak okunmali.</p>
          <div class="rq1-metrics">
            ${rq1MetricCard("Default", "MiniLM")}
            ${rq1MetricCard("Alt profile", "Multilingual")}
          </div>
        </article>
        <article class="rq1-card">
          <span class="rq1-label">Language Mix</span>
          <h4>Dil kesiti belirleyici</h4>
          <p>English sorgular ile Turkish sorgular ayni profile davranisini vermiyor; RQ3'te ana fark burada gorunuyor.</p>
          <div class="rq1-metrics">
            ${rq1MetricCard("English", String(rq3.language_counts?.en ?? "-"))}
            ${rq1MetricCard("Turkish", String(rq3.language_counts?.tr ?? "-"))}
          </div>
        </article>
      `;

      rq3OverviewEl.innerHTML = ["hybrid", "semantic"].map((method) => {
        const minilmRow = findProfileMethodRow(rq3.overall, "minilm", method);
        const multilingualRow = findProfileMethodRow(rq3.overall, "multilingual", method);
        if (!minilmRow || !multilingualRow) return "";
        const winner = (minilmRow.mean_ndcg_at_k ?? 0) >= (multilingualRow.mean_ndcg_at_k ?? 0) ? "minilm" : "multilingual";
        return `
          <article class="rq1-card">
            <span class="rq1-label">${escapeHtml(methodLabels[method])}</span>
            <h4>${escapeHtml(profileLabel(winner))} overall daha guclu</h4>
            <p>Ayni benchmark ailesinde iki profile'in genel ranking kalitesi burada karsilastiriliyor.</p>
            <div class="rq1-metrics">
              ${rq1MetricCard("MiniLM nDCG@5", Number(minilmRow.mean_ndcg_at_k).toFixed(3))}
              ${rq1MetricCard("Multi nDCG@5", Number(multilingualRow.mean_ndcg_at_k).toFixed(3))}
              ${rq1MetricCard("Winner", winner === "minilm" ? "MiniLM" : "Multi")}
            </div>
          </article>
        `;
      }).join("");

      rq3SliceEl.innerHTML = ["english_main", "cross_source", "tr_subset"].map((slice) => {
        const minilmHybrid = findProfileMethodRow(rq3.by_study_slice?.[slice], "minilm", "hybrid");
        const multilingualHybrid = findProfileMethodRow(rq3.by_study_slice?.[slice], "multilingual", "hybrid");
        const minilmSemantic = findProfileMethodRow(rq3.by_study_slice?.[slice], "minilm", "semantic");
        const multilingualSemantic = findProfileMethodRow(rq3.by_study_slice?.[slice], "multilingual", "semantic");
        if (!minilmHybrid || !multilingualHybrid || !minilmSemantic || !multilingualSemantic) return "";
        const winner = (minilmSemantic.mean_ndcg_at_k ?? 0) >= (multilingualSemantic.mean_ndcg_at_k ?? 0) ? "minilm" : "multilingual";
        return `
          <article class="rq1-card">
            <span class="rq1-label">${escapeHtml(formatStudySliceLabel(slice))}</span>
            <h4>${escapeHtml(profileLabel(winner))} semantic olarak onde</h4>
            <p>Bu study slice'ta model secimi retrieval kalitesini dogrudan degistiriyor.</p>
            <div class="rq1-metrics">
              ${rq1MetricCard("MiniLM semantic", Number(minilmSemantic.mean_ndcg_at_k).toFixed(3))}
              ${rq1MetricCard("Multi semantic", Number(multilingualSemantic.mean_ndcg_at_k).toFixed(3))}
              ${rq1MetricCard("MiniLM hybrid", Number(minilmHybrid.mean_ndcg_at_k).toFixed(3))}
            </div>
          </article>
        `;
      }).join("");

      rq3LanguageEl.innerHTML = ["en", "tr"].map((language) => {
        const minilmRow = findProfileMethodRow(rq3.by_language?.[language], "minilm", "semantic");
        const multilingualRow = findProfileMethodRow(rq3.by_language?.[language], "multilingual", "semantic");
        if (!minilmRow || !multilingualRow) return "";
        const winner = (minilmRow.mean_ndcg_at_k ?? 0) >= (multilingualRow.mean_ndcg_at_k ?? 0) ? "minilm" : "multilingual";
        return `
          <article class="rq1-card">
            <span class="rq1-label">${escapeHtml(formatLanguageLabel(language))}</span>
            <h4>${escapeHtml(profileLabel(winner))} daha uygun profile</h4>
            <p>Dil degistikce semantic model tercihi de degisiyor; RQ3'un ana pratik mesaji burada.</p>
            <div class="rq1-metrics">
              ${rq1MetricCard("MiniLM semantic", Number(minilmRow.mean_ndcg_at_k).toFixed(3))}
              ${rq1MetricCard("Multi semantic", Number(multilingualRow.mean_ndcg_at_k).toFixed(3))}
              ${rq1MetricCard("Preferred", winner === "minilm" ? "MiniLM" : "Multi")}
            </div>
          </article>
        `;
      }).join("");

      rq3BannerEl.innerHTML = `
        <strong>RQ3 ozet sonucu</strong>
        <p>Main English benchmarkta <b>MiniLM</b> daha guclu. Turkish alt kumede ise <b>multilingual</b> profile one cikiyor. Bu nedenle web uygulamasinda varsayilan English pipeline korunurken, iki dilli sorgular icin multilingual profile bilincli sekilde secilmeli.</p>
      `;
    }

    function renderRQ4Overview() {
      const rq4 = metadataPayload?.rq4;
      if (!rq4) return;

      const semanticLong = findMethodRow(rq4.by_length?.long, "semantic");
      const semanticShort = findMethodRow(rq4.by_length?.short, "semantic");
      const semanticNarrative = findMethodRow(rq4.by_style?.narrative, "semantic");
      const semanticMetadata = findMethodRow(rq4.by_style?.metadata_heavy, "semantic");
      const semanticRich = findMethodRow(rq4.by_term?.term_rich, "semantic");
      const semanticSparse = findMethodRow(rq4.by_term?.term_sparse, "semantic");
      const hybridSparse = findMethodRow(rq4.by_term?.term_sparse, "hybrid");
      const hybridMixed = findMethodRow(rq4.by_style?.mixed_structured, "hybrid");

      rq4CompositionEl.innerHTML = `
        <article class="rq1-card">
          <span class="rq1-label">Evaluation Scope</span>
          <h4>${escapeHtml(String(rq4.query_count ?? "-"))} English query</h4>
          <p>RQ4, mevcut thesis benchmarkini belge kalitesi perspektifinden yeniden okuyor.</p>
          <div class="rq1-metrics">
            ${rq1MetricCard("English main", String(rq4.study_slice_counts?.english_main ?? "-"))}
            ${rq1MetricCard("Cross-source", String(rq4.study_slice_counts?.cross_source ?? "-"))}
          </div>
        </article>
        <article class="rq1-card">
          <span class="rq1-label">Bucket Rules</span>
          <h4>Kalite kirilimlari</h4>
          <p>Uzunluk, terim zenginligi ve anlatim stili birlikte semantic sinyalin ne kadar guclu tasindigini olcuyor.</p>
          <div class="rq1-metrics">
            ${rq1MetricCard("Short <= ", String(rq4.thresholds?.length_word_thresholds?.short_max ?? "-"))}
            ${rq1MetricCard("Medium <= ", String(rq4.thresholds?.length_word_thresholds?.medium_max ?? "-"))}
            ${rq1MetricCard("Sparse <= ", String(rq4.thresholds?.term_thresholds?.sparse_max ?? "-"))}
          </div>
        </article>
        <article class="rq1-card">
          <span class="rq1-label">Corpus Mix</span>
          <h4>Dagilim asimetrik degil</h4>
          <p>Korpustaki short, medium, long ve term bucket oranlari birbirine yakin; bu da RQ4 gozlemlerini tek bir uc gruba baglamiyor.</p>
          <div class="rq1-metrics">
            ${rq1MetricCard("Metadata heavy", `${Math.round((rq4.distribution?.description_style?.metadata_heavy ?? 0) * 100)}%`)}
            ${rq1MetricCard("Mixed", `${Math.round((rq4.distribution?.description_style?.mixed_structured ?? 0) * 100)}%`)}
            ${rq1MetricCard("Narrative", `${Math.round((rq4.distribution?.description_style?.narrative ?? 0) * 100)}%`)}
          </div>
        </article>
      `;

      rq4StyleEl.innerHTML = `
        <article class="rq1-card">
          <span class="rq1-label">Style Signal</span>
          <h4>Narrative aciklamalar semantic icin daha guclu</h4>
          <p>Salt metadata benzeri aciklamalar semantic sinyali zayiflatirken, anlatimsal ve aciklayici metinler retrieval'i daha iyi tasiyor.</p>
          <div class="rq1-metrics">
            ${semanticNarrative ? rq1MetricCard("Narrative Hit@5", Number(semanticNarrative.hit_rate_at_k).toFixed(3)) : ""}
            ${semanticMetadata ? rq1MetricCard("Metadata Hit@5", Number(semanticMetadata.hit_rate_at_k).toFixed(3)) : ""}
            ${hybridMixed ? rq1MetricCard("Hybrid mixed Top1", Number(hybridMixed.top1_rate).toFixed(3)) : ""}
          </div>
        </article>
      `;

      rq4LengthEl.innerHTML = `
        <article class="rq1-card">
          <span class="rq1-label">Length Signal</span>
          <h4>Daha uzun aciklamalar daha kolay toparlaniyor</h4>
          <p>Semantic retrieval, uzun aciklamalarda daha fazla baglam yakaliyor. Kisa kayitlarda ise hem semantic hem hybrid daha sinirli kaliyor.</p>
          <div class="rq1-metrics">
            ${semanticLong ? rq1MetricCard("Long Hit@5", Number(semanticLong.hit_rate_at_k).toFixed(3)) : ""}
            ${semanticShort ? rq1MetricCard("Short Hit@5", Number(semanticShort.hit_rate_at_k).toFixed(3)) : ""}
            ${semanticLong ? rq1MetricCard("Long Top1", Number(semanticLong.top1_rate).toFixed(3)) : ""}
          </div>
        </article>
      `;

      rq4TermEl.innerHTML = `
        <article class="rq1-card">
          <span class="rq1-label">Term Signal</span>
          <h4>Terim zenginligi semantic temsili destekliyor</h4>
          <p>Icerik terimleri arttikca semantic eslesme gucleniyor. Zayif aciklamalarda hybrid yararli bir geri donus mekanizmasi oluyor.</p>
          <div class="rq1-metrics">
            ${semanticRich ? rq1MetricCard("Rich Hit@5", Number(semanticRich.hit_rate_at_k).toFixed(3)) : ""}
            ${semanticSparse ? rq1MetricCard("Sparse Hit@5", Number(semanticSparse.hit_rate_at_k).toFixed(3)) : ""}
            ${hybridSparse ? rq1MetricCard("Hybrid sparse Hit@5", Number(hybridSparse.hit_rate_at_k).toFixed(3)) : ""}
          </div>
        </article>
      `;

      rq4BannerEl.innerHTML = `
        <strong>RQ4 ozet sonucu</strong>
        <p>Semantic kalite yalnizca model secimiyle belirlenmiyor. Daha uzun, daha anlatimsal ve daha terim zengini aciklamalar retrieval'i guclendiriyor; zayif kayitlarda ise <b>hybrid</b> daha guvenli bir fallback oluyor.</p>
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
              ${result.quality_flag_count ? `<span class="${qualityChipClass(result.quality_status)}">${escapeHtml(result.quality_flag_summary || "Flag var")}</span>` : ""}
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

    function renderRQ2Live(payload) {
      const rq2 = payload.rq2;
      if (!rq2) {
        rq2LivePanelEl.hidden = true;
        return;
      }

      rq2LivePanelEl.hidden = false;
      if (!rq2.enabled) {
        rq2LiveNoteEl.textContent = "Canli cross-source yorum icin hedef kaynagi tek platforma indir.";
        rq2GuidanceEl.innerHTML = `
          <strong>RQ2 canli modu hazir</strong>
          <p>Source filtresini <b>Kaggle</b> ya da <b>Hugging Face</b> olarak sec. Boylece sistem, sorguyu diger platformdan bu hedef kaynaga kopruleme senaryosu olarak yorumlayip ilgili yon benchmarki ile karsilastirir.</p>
        `;
        rq2LiveGridEl.innerHTML = `
          <article class="rq1-card">
            <span class="rq1-label">How To Use</span>
            <h3>Target source sec</h3>
            <p><b>Kaggle</b> secildiginde canli panel <b>Hugging Face -> Kaggle</b> yonunu, <b>Hugging Face</b> secildiginde ise <b>Kaggle -> Hugging Face</b> yonunu baz alir.</p>
            <div class="rq1-metrics">
              ${rq1MetricCard("Current mode", "All sources")}
              ${rq1MetricCard("Needed", "Single source")}
            </div>
          </article>
        `;
        return;
      }

      rq2LiveNoteEl.textContent = `Secilen hedef kaynak ${rq2.target_source_label}. Bu kurulum RQ2 benchmarkindaki ${rq2.direction_label} yonune en yakin canli senaryodur.`;
      rq2GuidanceEl.innerHTML = `
        <strong>Cross-source yorum</strong>
        <p>${escapeHtml(rq2.guidance_text || "")}</p>
      `;

      rq2LiveGridEl.innerHTML = (rq2.snapshots || []).map((snapshot) => {
        const top = snapshot.top_result;
        const benchmark = snapshot.benchmark_metrics;
        return `
          <article class="rq1-card">
            <span class="rq1-label">${escapeHtml(methodBadge(snapshot.method))}</span>
            <h3>${escapeHtml(methodLabels[snapshot.method] || snapshot.method)}</h3>
            <p>${escapeHtml(methodRole(snapshot.method))}</p>
            <div class="rq1-metrics">
              ${benchmark ? rq1MetricCard("Benchmark nDCG@5", Number(benchmark.mean_ndcg_at_k).toFixed(3)) : ""}
              ${benchmark ? rq1MetricCard("Benchmark Bridge@5", Number(benchmark.mean_hit_rate_at_k).toFixed(3)) : ""}
              ${top ? rq1MetricCard("Live top score", formatScore(top.score)) : ""}
            </div>
            <p><strong>Top-1:</strong> ${escapeHtml(top?.title || "Sonuc yok")}</p>
            <p>${escapeHtml(top?.source ? top.source.toUpperCase() : "")}${top?.ref ? ` / ${top.ref}` : ""}</p>
          </article>
        `;
      }).join("");
    }

    function renderRQ3Live(payload) {
      const rq3 = payload.rq3;
      if (!rq3) {
        rq3LivePanelEl.hidden = true;
        return;
      }

      rq3LivePanelEl.hidden = false;
      rq3LiveNoteEl.textContent = `Algilanan sorgu dili ${rq3.language_label}. Aktif benchmark kesiti: ${rq3.primary_slice_label}.`;
      rq3GuidanceEl.innerHTML = `
        <strong>Model yorumu</strong>
        <p>${escapeHtml(rq3.guidance_text || "")}</p>
      `;

      rq3LiveGridEl.innerHTML = (rq3.snapshots || []).map((snapshot) => {
        const top = snapshot.top_result;
        const primary = snapshot.primary_metrics;
        const language = snapshot.language_metrics;
        return `
          <article class="rq1-card">
            <span class="rq1-label">${escapeHtml(snapshot.is_selected ? "Active profile" : "Alt profile")}</span>
            <h3>${escapeHtml(snapshot.profile_label || snapshot.profile)}</h3>
            <p>${escapeHtml(snapshot.profile_description || "")}</p>
            <div class="rq1-metrics">
              ${primary ? rq1MetricCard("Slice nDCG@5", Number(primary.mean_ndcg_at_k).toFixed(3)) : ""}
              ${language ? rq1MetricCard("Lang nDCG@5", Number(language.mean_ndcg_at_k).toFixed(3)) : ""}
              ${top ? rq1MetricCard("Live top score", formatScore(top.score)) : ""}
            </div>
            <p><strong>Top-1:</strong> ${escapeHtml(top?.title || "Sonuc yok")}</p>
            <p>${escapeHtml(top?.source ? top.source.toUpperCase() : "")}${top?.ref ? ` / ${top.ref}` : ""}</p>
          </article>
        `;
      }).join("");
    }

    function renderRQ4Live(payload) {
      const rq4 = payload.rq4;
      if (!rq4) {
        rq4LivePanelEl.hidden = true;
        return;
      }

      rq4LivePanelEl.hidden = false;
      rq4LiveNoteEl.textContent = `Canli kalite yorumu ${rq4.compare_method_label} methodu etrafinda kuruluyor. Algilanan sorgu dili: ${rq4.language_label}.`;
      rq4GuidanceEl.innerHTML = `
        <strong>Belge kalite yorumu</strong>
        <p>${escapeHtml(rq4.guidance_text || "")}</p>
      `;

      rq4LiveGridEl.innerHTML = (rq4.snapshots || []).map((snapshot) => {
        const top = snapshot.top_result;
        const lengthMetrics = snapshot.length_metrics;
        const styleMetrics = snapshot.style_metrics;
        const termMetrics = snapshot.term_metrics;
        return `
          <article class="rq1-card">
            <span class="rq1-label">${escapeHtml(methodLabels[snapshot.method] || snapshot.method)}</span>
            <h3>${escapeHtml(top?.title || "Sonuc yok")}</h3>
            <p>${escapeHtml(snapshot.description_style_label || "-")} / ${escapeHtml(snapshot.length_bucket_label || "-")} / ${escapeHtml(snapshot.term_bucket_label || "-")}</p>
            <div class="rq1-metrics">
              ${lengthMetrics ? rq1MetricCard("Len Hit@5", Number(lengthMetrics.hit_rate_at_k).toFixed(3)) : ""}
              ${styleMetrics ? rq1MetricCard("Style Hit@5", Number(styleMetrics.hit_rate_at_k).toFixed(3)) : ""}
              ${termMetrics ? rq1MetricCard("Term Hit@5", Number(termMetrics.hit_rate_at_k).toFixed(3)) : ""}
            </div>
            <p><strong>Top-1:</strong> ${escapeHtml(top?.source ? top.source.toUpperCase() : "")}${top?.ref ? ` / ${top.ref}` : ""}</p>
          </article>
        `;
      }).join("");
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
        ["Quality status", result.quality_flag_summary || "Flag yok"],
        ["Quality flag count", String(result.quality_flag_count ?? 0)]
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
      modalFlags.innerHTML = renderQualityFlags(result.quality_flag_details);
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
      renderRQ2Overview();
      renderRQ3Overview();
      renderRQ4Overview();
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

    tabButtons.forEach((button) => {
      button.addEventListener("click", () => {
        activateTab(button.dataset.tab);
        if (button.dataset.tab === "search") {
          queryInput.focus();
        }
      });
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
        renderRQ2Live(payload);
        renderRQ3Live(payload);
        renderRQ4Live(payload);
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
        activateTab("search");
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


def group_rows_by(rows, key_name):
    grouped = {}
    for row in rows:
        key = row.get(key_name)
        if not key:
            continue
        grouped.setdefault(key, []).append(row)
    return grouped


def detect_query_language(query: str) -> str:
    text = (query or "").lower()
    if any(char in text for char in "çğıöşü"):
        return "tr"
    turkish_markers = {
        "ve",
        "bir",
        "icin",
        "için",
        "ile",
        "olan",
        "veri",
        "seti",
        "yorum",
        "turkce",
        "türkçe",
    }
    tokens = {token for token in text.replace(",", " ").replace(".", " ").split() if token}
    return "tr" if tokens.intersection(turkish_markers) else "en"


def estimate_content_term_count(text: str) -> int:
    tokens = {
        token.lower()
        for token in CONTENT_TERM_RE.findall(text or "")
        if token.lower() not in CONTENT_STOPWORDS
    }
    return len(tokens)


def infer_length_bucket(word_count: int, thresholds: dict) -> str:
    short_max = thresholds.get("short_max", 20)
    medium_max = thresholds.get("medium_max", 52)
    if word_count <= short_max:
        return "short"
    if word_count <= medium_max:
        return "medium"
    return "long"


def infer_term_bucket(term_count: int, thresholds: dict) -> str:
    sparse_max = thresholds.get("sparse_max", 10)
    moderate_max = thresholds.get("moderate_max", 28)
    if term_count <= sparse_max:
        return "term_sparse"
    if term_count <= moderate_max:
        return "term_moderate"
    return "term_rich"


def infer_description_style(description: str, keywords: list, word_count: int) -> str:
    text = description or ""
    lowered = text.lower()
    colon_count = text.count(":")
    bullet_signals = text.count("- ") + text.count("•") + lowered.count("keywords:") + lowered.count("use cases:") + lowered.count("columns:")
    keyword_count = len(keywords or [])
    if word_count <= 40 and (keyword_count >= 3 or colon_count >= 3):
        return "metadata_heavy"
    if bullet_signals >= 2 or colon_count >= 4:
        return "mixed_structured"
    if word_count >= 80:
        return "narrative"
    return "mixed_structured" if keyword_count else "narrative"


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for value in values:
        if not value or value in seen:
            continue
        ordered.append(value)
        seen.add(value)
    return ordered


def build_quality_flag_payload(item: dict) -> dict:
    description = item.get("description") or item.get("text") or ""
    keywords = item.get("keywords") or []
    word_count = int(item.get("description_len_words") or len(description.split()))
    term_count = estimate_content_term_count(description)
    raw_flags = dedupe_preserve_order(list(item.get("quality_flags") or []))

    style = infer_description_style(description, keywords, word_count) if description else None
    length_bucket = infer_length_bucket(
        word_count,
        {"short_max": 20, "medium_max": 52},
    ) if word_count else None
    term_bucket = infer_term_bucket(
        term_count,
        {"sparse_max": 10, "moderate_max": 28},
    ) if term_count or description else None

    inferred_flags = []
    if not description.strip():
        inferred_flags.append("empty")
    if length_bucket == "short" and not any(flag in raw_flags for flag in ("short_description", "too_short")):
        inferred_flags.append("short_context")
    if style == "metadata_heavy":
        inferred_flags.append("metadata_heavy")
    if term_bucket == "term_sparse":
        inferred_flags.append("term_sparse")
    if ("keyword_only" in raw_flags or style == "metadata_heavy") and length_bucket == "short":
        inferred_flags.append("low_information")
    elif style == "metadata_heavy" and term_bucket == "term_sparse":
        inferred_flags.append("low_information")

    all_flags = dedupe_preserve_order(raw_flags + inferred_flags)
    details = []
    severity_rank = {"info": 0, "warning": 1, "risk": 2}
    top_severity = "info"

    for code in all_flags:
        meta = QUALITY_FLAG_META.get(code)
        if not meta:
            meta = {
                "label": code.replace("_", " ").title(),
                "severity": "warning",
                "message": "Kayit bu kalite bayragi ile isaretlenmis.",
            }
        severity = meta["severity"]
        if severity_rank[severity] > severity_rank[top_severity]:
            top_severity = severity
        details.append(
            {
                "code": code,
                "label": meta["label"],
                "severity": severity,
                "message": meta["message"],
                "origin": "stored" if code in raw_flags else "inferred",
            }
        )

    summary_map = {
        "info": "Temiz kayit",
        "warning": "Dikkat gerekiyor",
        "risk": "Riskli kayit",
    }

    return {
        "quality_flag_codes": all_flags,
        "quality_flag_details": details,
        "quality_flag_count": len(details),
        "quality_status": top_severity if details else "info",
        "quality_flag_summary": summary_map[top_severity] if details else "Flag yok",
        "description_style": style,
        "length_bucket": length_bucket,
        "term_bucket": term_bucket,
    }


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
        "rq2": load_rq2_metadata(),
        "rq3": load_rq3_metadata(),
        "rq4": load_rq4_metadata(),
    }


def load_rq1_metadata():
    overall = load_json_file(RQ1_SUMMARY_PATH, [])
    by_query_style_rows = load_json_file(RQ1_QUERY_STYLE_PATH, [])
    by_benchmark_rows = load_json_file(RQ1_BENCHMARK_PATH, [])
    return {
        "overall": overall,
        "by_query_style": group_rows_by(by_query_style_rows, "query_style"),
        "by_benchmark": by_benchmark_rows,
    }


def load_rq2_metadata():
    overall = load_json_file(RQ2_SUMMARY_PATH, [])
    by_direction_rows = load_json_file(RQ2_DIRECTION_PATH, [])
    by_topic_rows = load_json_file(RQ2_TOPIC_PATH, [])
    direction_counts = {}
    for row in by_direction_rows:
        direction = row.get("direction")
        if not direction or direction in direction_counts:
            continue
        direction_counts[direction] = row.get("queries", 0)

    topic_counts = {}
    for row in by_topic_rows:
        topic = row.get("topic")
        if not topic or topic in topic_counts:
            continue
        topic_counts[topic] = row.get("queries", 0)

    largest_topic = None
    smallest_topic = None
    if topic_counts:
        sorted_topics = sorted(topic_counts.items(), key=lambda item: (item[1], item[0]))
        smallest_topic = sorted_topics[0][0]
        largest_topic = sorted_topics[-1][0]

    return {
        "overall": overall,
        "by_direction": group_rows_by(by_direction_rows, "direction"),
        "by_topic": group_rows_by(by_topic_rows, "topic"),
        "query_count": overall[0].get("queries") if overall else 0,
        "direction_counts": direction_counts,
        "topic_counts": topic_counts,
        "largest_topic": largest_topic,
        "smallest_topic": smallest_topic,
    }


def load_rq3_metadata():
    overall = load_json_file(RQ3_SUMMARY_PATH, [])
    by_language_rows = load_json_file(RQ3_LANGUAGE_PATH, [])
    by_benchmark_rows = load_json_file(RQ3_BENCHMARK_PATH, [])
    by_study_slice_rows = load_json_file(RQ3_STUDY_SLICE_PATH, [])

    language_counts = {}
    for row in by_language_rows:
        language = row.get("language")
        if not language or language in language_counts:
            continue
        language_counts[language] = row.get("queries", 0)

    study_slice_counts = {}
    for row in by_study_slice_rows:
        study_slice = row.get("study_slice")
        if not study_slice or study_slice in study_slice_counts:
            continue
        study_slice_counts[study_slice] = row.get("queries", 0)

    benchmark_counts = {}
    for row in by_benchmark_rows:
        benchmark = row.get("benchmark")
        if not benchmark or benchmark in benchmark_counts:
            continue
        benchmark_counts[benchmark] = row.get("queries", 0)

    return {
        "overall": overall,
        "by_language": group_rows_by(by_language_rows, "language"),
        "by_benchmark": group_rows_by(by_benchmark_rows, "benchmark"),
        "by_study_slice": group_rows_by(by_study_slice_rows, "study_slice"),
        "query_count": overall[0].get("queries") if overall else 0,
        "language_counts": language_counts,
        "study_slice_counts": study_slice_counts,
        "benchmark_counts": benchmark_counts,
    }


def load_rq4_metadata():
    details = load_json_file(RQ4_DETAILS_PATH, [])
    by_style_rows = load_json_file(RQ4_STYLE_PATH, [])
    by_length_rows = load_json_file(RQ4_LENGTH_PATH, [])
    by_term_rows = load_json_file(RQ4_TERM_PATH, [])
    corpus_distribution = load_json_file(RQ4_CORPUS_DISTRIBUTION_PATH, {})

    unique_queries = set()
    study_slice_counts = {}
    for row in details:
        query_id = row.get("query_id")
        study_slice = row.get("study_slice")
        if query_id:
            unique_queries.add(query_id)
        if query_id and study_slice:
            study_slice_counts.setdefault(study_slice, set()).add(query_id)

    distribution = {}
    for row in corpus_distribution.get("distribution", []):
        feature = row.get("feature")
        bucket = row.get("bucket")
        share = row.get("share")
        if not feature or not bucket:
            continue
        distribution.setdefault(feature, {})[bucket] = share

    return {
        "query_count": len(unique_queries),
        "study_slice_counts": {key: len(value) for key, value in study_slice_counts.items()},
        "by_style": group_rows_by(by_style_rows, "description_style"),
        "by_length": group_rows_by(by_length_rows, "length_bucket"),
        "by_term": group_rows_by(by_term_rows, "term_bucket"),
        "distribution": distribution,
        "thresholds": corpus_distribution.get("thresholds", {}),
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


def find_profile_method_row(rows, profile_key, method):
    for row in rows:
        if row.get("profile") == profile_key and row.get("method") == method:
            return row
    return None


def get_other_profile_key(profile_key):
    for candidate in PROFILE_ORDER:
        if candidate != profile_key:
            return candidate
    return profile_key


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


def build_rq2_payload(profile_key, source_filter, top_k, method_results):
    if source_filter == "all":
        return {
            "enabled": False,
            "source_filter": source_filter,
            "top_k": top_k,
        }

    rq2_data = (AppState.metadata or {}).get("rq2") or {}
    direction = (
        "huggingface_to_kaggle"
        if source_filter == "kaggle"
        else "kaggle_to_huggingface"
    )
    direction_rows = (rq2_data.get("by_direction") or {}).get(direction, [])
    recommended = max(
        direction_rows,
        key=lambda row: row.get("mean_ndcg_at_k", 0.0),
        default=None,
    )

    guidance = [
        f"Hedef kaynak `{source_filter}` oldugu icin bu canli arama, RQ2 benchmarkindaki `{DIRECTION_LABELS.get(direction, direction)}` yonune en yakin senaryo."
    ]
    if recommended:
        guidance.append(
            f"Bu yonde en guclu method `{METHOD_LABELS.get(recommended['method'], recommended['method'])}` "
            f"(nDCG@5={recommended['mean_ndcg_at_k']:.3f}, Bridge@5={recommended['mean_hit_rate_at_k']:.3f})."
        )

    semantic_row = find_method_row(direction_rows, "semantic")
    bm25_row = find_method_row(direction_rows, "bm25")
    hybrid_row = find_method_row(direction_rows, "hybrid")
    if semantic_row and bm25_row:
        guidance.append(
            f"Semantic retrieval, BM25'e gore daha guclu cross-source kopru kuruyor "
            f"(semantic Bridge@5={semantic_row['mean_hit_rate_at_k']:.3f}, BM25={bm25_row['mean_hit_rate_at_k']:.3f})."
        )
    if hybrid_row and semantic_row:
        guidance.append(
            f"Hybrid uygulamadaki en guclu sistem olarak korunabilir; ancak RQ2'nin dogrudan cevabi semantic retrieval performansidir."
        )
    if profile_key != DEFAULT_WEB_PROFILE:
        guidance.append(
            "Not: RQ2 benchmark metrikleri tezin ana Ingilizce MiniLM profiline aittir."
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
                "benchmark_metrics": find_method_row(direction_rows, method),
            }
        )

    return {
        "enabled": True,
        "direction": direction,
        "direction_label": DIRECTION_LABELS.get(direction, direction),
        "target_source": source_filter,
        "target_source_label": "Kaggle" if source_filter == "kaggle" else "Hugging Face",
        "recommended_method": recommended.get("method") if recommended else None,
        "guidance_text": " ".join(guidance),
        "top_k": top_k,
        "snapshots": snapshots,
    }


def build_rq3_payload(query, profile_key, selected_method, source_filter, top_k, query_plan, method_results):
    rq3_data = (AppState.metadata or {}).get("rq3") or {}
    language = detect_query_language(query)
    compare_method = selected_method if selected_method in {"semantic", "hybrid"} else "hybrid"
    primary_slice = "tr_subset" if language == "tr" else ("cross_source" if source_filter != "all" else "english_main")
    alternative_profile_key = get_other_profile_key(profile_key)

    primary_rows = (rq3_data.get("by_study_slice") or {}).get(primary_slice, [])
    language_rows = (rq3_data.get("by_language") or {}).get(language, [])
    current_primary = find_profile_method_row(primary_rows, profile_key, compare_method)
    alternative_primary = find_profile_method_row(primary_rows, alternative_profile_key, compare_method)
    current_language = find_profile_method_row(language_rows, profile_key, compare_method)
    alternative_language = find_profile_method_row(language_rows, alternative_profile_key, compare_method)

    guidance = []
    if compare_method != selected_method:
        guidance.append(
            "RQ3 benchmarki BM25'i kapsamadigi icin model etkisi karsilastirmasi hybrid katmani uzerinden yapiliyor."
        )

    guidance.append(
        f"Algilanan sorgu dili `{LANGUAGE_LABELS.get(language, language)}`. Bu nedenle birincil karsilastirma kesiti `{STUDY_SLICE_LABELS.get(primary_slice, primary_slice)}` olarak secildi."
    )

    recommended_profile = profile_key
    if current_primary and alternative_primary:
        current_score = current_primary.get("mean_ndcg_at_k", 0.0)
        alternative_score = alternative_primary.get("mean_ndcg_at_k", 0.0)
        recommended_profile = profile_key if current_score >= alternative_score else alternative_profile_key
        guidance.append(
            f"{METHOD_LABELS.get(compare_method, compare_method)} icin bu kesitte `{get_profile(recommended_profile).label}` daha guclu gorunuyor "
            f"(aktif profile nDCG@5={current_score:.3f}, alternatif nDCG@5={alternative_score:.3f})."
        )

    if current_language and alternative_language and language == "tr":
        guidance.append(
            f"Turkish sorgularda multilingual profile daha guclu beklenir "
            f"(MiniLM nDCG@5={find_profile_method_row(language_rows, 'minilm', compare_method).get('mean_ndcg_at_k', 0.0):.3f}, "
            f"Multilingual nDCG@5={find_profile_method_row(language_rows, 'multilingual', compare_method).get('mean_ndcg_at_k', 0.0):.3f})."
        )
    elif current_language and alternative_language:
        guidance.append(
            f"English agirlikli sorgularda MiniLM ana profile olarak daha guclu kalir "
            f"(MiniLM nDCG@5={find_profile_method_row(language_rows, 'minilm', compare_method).get('mean_ndcg_at_k', 0.0):.3f}, "
            f"Multilingual nDCG@5={find_profile_method_row(language_rows, 'multilingual', compare_method).get('mean_ndcg_at_k', 0.0):.3f})."
        )

    selected_rows = method_results.get(compare_method) or []
    alternative_rows = execute_search_method(
        compare_method,
        query,
        alternative_profile_key,
        source_filter,
        top_k,
        query_plan,
    )

    snapshots = []
    for candidate_profile_key, rows, is_selected in (
        (profile_key, selected_rows, True),
        (alternative_profile_key, alternative_rows, False),
    ):
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
        profile = get_profile(candidate_profile_key)
        snapshots.append(
            {
                "profile": candidate_profile_key,
                "profile_label": profile.label,
                "profile_description": profile.description,
                "is_selected": is_selected,
                "top_result": top_result,
                "primary_metrics": find_profile_method_row(primary_rows, candidate_profile_key, compare_method),
                "language_metrics": find_profile_method_row(language_rows, candidate_profile_key, compare_method),
            }
        )

    return {
        "enabled": True,
        "selected_profile": profile_key,
        "alternative_profile": alternative_profile_key,
        "language": language,
        "language_label": LANGUAGE_LABELS.get(language, language),
        "primary_slice": primary_slice,
        "primary_slice_label": STUDY_SLICE_LABELS.get(primary_slice, primary_slice),
        "compare_method": compare_method,
        "recommended_profile": recommended_profile,
        "recommended_profile_label": get_profile(recommended_profile).label,
        "guidance_text": " ".join(guidance),
        "top_k": top_k,
        "snapshots": snapshots,
    }


def build_rq4_snapshot(method, rows, rq4_data):
    top = rows[0] if rows else None
    top_result = None
    description_style = None
    length_bucket = None
    term_bucket = None
    length_metrics = None
    style_metrics = None
    term_metrics = None

    if top is not None:
        _, item = top
        description = item.get("description") or item.get("text") or ""
        keywords = item.get("keywords") or []
        word_count = int(item.get("description_len_words") or len(description.split()))
        term_count = estimate_content_term_count(description)
        threshold_block = rq4_data.get("thresholds") or {}
        length_thresholds = threshold_block.get("length_word_thresholds") or {}
        term_thresholds = threshold_block.get("term_thresholds") or {}
        length_bucket = infer_length_bucket(word_count, length_thresholds)
        term_bucket = infer_term_bucket(term_count, term_thresholds)
        description_style = infer_description_style(description, keywords, word_count)
        length_metrics = find_method_row((rq4_data.get("by_length") or {}).get(length_bucket, []), method)
        style_metrics = find_method_row((rq4_data.get("by_style") or {}).get(description_style, []), method)
        term_metrics = find_method_row((rq4_data.get("by_term") or {}).get(term_bucket, []), method)
        top_result = {
            "title": item.get("title"),
            "source": item.get("source"),
            "ref": item.get("ref"),
            "word_count": word_count,
            "term_count": term_count,
        }

    return {
        "method": method,
        "top_result": top_result,
        "description_style": description_style,
        "description_style_label": DESCRIPTION_STYLE_LABELS.get(description_style, description_style),
        "length_bucket": length_bucket,
        "length_bucket_label": LENGTH_BUCKET_LABELS.get(length_bucket, length_bucket),
        "term_bucket": term_bucket,
        "term_bucket_label": TERM_BUCKET_LABELS.get(term_bucket, term_bucket),
        "length_metrics": length_metrics,
        "style_metrics": style_metrics,
        "term_metrics": term_metrics,
    }


def build_rq4_payload(query, profile_key, selected_method, method_results):
    rq4_data = (AppState.metadata or {}).get("rq4") or {}
    language = detect_query_language(query)
    compare_method = selected_method if selected_method in {"semantic", "hybrid"} else "hybrid"

    guidance = []
    if compare_method != selected_method:
        guidance.append(
            "RQ4 benchmarki belge kalite etkisini semantic ve hybrid katmanlari uzerinden okudugu icin BM25 seciminde hybrid fallback kullaniliyor."
        )
    if language != "en":
        guidance.append(
            "RQ4 benchmarki English query'ler ile uretilmistir; Turkish sorgularda gosterilen yorum belge kalite sezgisidir, dogrudan benchmark esdegeri degildir."
        )
    if profile_key != DEFAULT_WEB_PROFILE:
        guidance.append(
            "RQ4 bucket metrikleri tezin ana MiniLM profile'ina aittir; yine de canli panel secili profile'in getirdigi top sonucu kalite sinyali olarak yorumlar."
        )

    snapshots = []
    for method in ("semantic", "hybrid"):
        snapshot = build_rq4_snapshot(method, method_results.get(method) or [], rq4_data)
        snapshots.append(snapshot)

    active_snapshot = next((snapshot for snapshot in snapshots if snapshot.get("method") == compare_method), None)
    if active_snapshot and active_snapshot.get("top_result"):
        guidance.append(
            f"Aktif top sonuc `{active_snapshot['description_style_label']}` / `{active_snapshot['length_bucket_label']}` / `{active_snapshot['term_bucket_label']}` sinyali veriyor."
        )
        if active_snapshot.get("length_bucket") == "short" or active_snapshot.get("term_bucket") == "term_sparse":
            guidance.append(
                "Kisa ya da terimce zayif aciklamalarda hybrid daha guvenli bir fallback olarak tutulmali."
            )
        elif active_snapshot.get("description_style") == "narrative" or active_snapshot.get("term_bucket") == "term_rich":
            guidance.append(
                "Anlatimsal ve terimce zengin aciklamalar semantic retrieval icin daha saglam sinyal tasiyor."
            )

    return {
        "enabled": True,
        "language": language,
        "language_label": LANGUAGE_LABELS.get(language, language),
        "compare_method": compare_method,
        "compare_method_label": METHOD_LABELS.get(compare_method, compare_method),
        "guidance_text": " ".join(guidance),
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
            quality_payload = build_quality_flag_payload(item)
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
                    **quality_payload,
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
                "rq2": build_rq2_payload(profile.key, source_filter, top_k, method_results),
                "rq3": build_rq3_payload(query, profile.key, method, source_filter, top_k, query_plan, method_results),
                "rq4": build_rq4_payload(query, profile.key, method, method_results),
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
