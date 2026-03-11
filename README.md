# PP Market Intelligence

A local RAG (Retrieval-Augmented Generation) system that makes public company filings queryable in plain English. Ask natural language questions against any company's 10-K or 10-Q filings and get precise, grounded answers — with year-over-year delta analysis, cross-company comparisons, and a Bloomberg-style browser interface.

Built from scratch with no managed vector database, no LangChain, and no abstraction layers. Every component is hand-rolled and transparent.

> **Current data source:** SEC EDGAR (public filings, free, no API key required).
> Future iterations could extend to private market filings, earnings call transcripts, news sources, and alternative data streams — the pipeline is data-source agnostic.

---

## What Problem This Solves

Public company 10-K filings average 150+ pages. Analysts spend hours extracting the signal that matters — risk posture shifts, strategic pivots, year-over-year changes in investment priorities. Most retail investors never read them at all.

PP Market Intelligence makes any filing instantly queryable. Ask a plain-English question, get a grounded answer cited to the exact section of the filing. Compare two companies on any topic. Track how a company's position on something changed between years.

---

## Live Demo

The interface runs locally against a FastAPI backend exposed via Cloudflare tunnel. Three core modes:

**Query** — ask anything about any ingested filing
> *"What are Uber's main risk factors?"*
> *"How did international revenue perform?"*
> *"What does management say about their competitive moat?"*

**Compare** — two companies, one topic, side-by-side
> *UBER vs LYFT on autonomous vehicles*
> *MSFT vs GOOGL on AI investment strategy*

**Delta** — how did a company's position on a topic change year-over-year?
> *UBER: autonomous vehicles, 2024 → 2026*
> Surfaced: $1.4B swing in Aurora investment valuation, disappearance of AV commercialisation language, emergence of AI disintermediation as the new competitive threat

---

## Architecture

```
SEC EDGAR API (free, public)
        │
        ▼
   ingest.py
        │
        ├── iXBRL viewer unwrapper
        ├── HTML parser (BeautifulSoup + lxml)
        └── Section-aware chunker
                splits on Item 1., Item 1A., Item 7., etc.
                each chunk tagged with canonical section label
                │
                ▼
        sentence-transformers
        (all-MiniLM-L6-v2, runs locally, no API cost)
                │
                ▼
        vectorstore.py
        (numpy cosine similarity, file-backed .jsonl + .npy)
                │
                ▼
   query.py ──► top-k retrieval ──► Claude Haiku ──► answer
                │
                ▼
   server.py (FastAPI, port 8000)
                │
                ▼
   frontend.html (browser UI, connects via Cloudflare tunnel)
```

---

## File Structure

```
pp-market-intelligence/
├── main.py              # CLI — ingest, list, ask, compare, delta commands
├── ingest.py            # EDGAR fetcher, iXBRL unwrapper, section chunker, embedder
├── query.py             # Retrieval, context builder, Claude Haiku generation
├── vectorstore.py       # Custom numpy vector store — no external DB required
├── server.py            # FastAPI HTTP server (port 8000) — 6 endpoints
├── frontend.html        # Browser UI — Query, Compare, Delta tabs + ingest panel
├── requirements.txt     # Python dependencies
├── .env                 # ANTHROPIC_API_KEY (not committed)
└── data/
    └── vectorstore/
        ├── documents.jsonl   # Chunks + metadata
        └── embeddings.npy    # Embedding vectors
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **Custom numpy vector store** | ChromaDB incompatible with Python 3.14; hand-rolled store is simpler and more transparent |
| **Local sentence-transformers** | Zero API cost, works offline, sufficient recall quality for long-form filing text |
| **Section-aware chunking** | 10-Ks have canonical Item X. headers — chunking by section preserves semantic coherence |
| **iXBRL unwrapping** | SEC serves filings inside an inline viewer wrapper; ingest.py detects and strips this |
| **Claude Haiku for generation** | Lowest-cost Anthropic model; grounded Q&A with retrieved context doesn't require a frontier model |
| **No LangChain / LlamaIndex** | Full control over every component; easier to extend and debug |
| **FastAPI + Cloudflare tunnel** | Exposes local server to browser without deployment |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/list` | All ingested filings with metadata |
| `POST` | `/ingest` | Ingest tickers from EDGAR |
| `POST` | `/ask` | Natural language question |
| `POST` | `/compare` | Side-by-side comparison of two tickers |
| `POST` | `/delta` | Year-over-year change analysis |

---

## Setup

```bash
git clone https://github.com/pratyushpaul93-coder/pp-market-intelligence
cd pp-market-intelligence
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
echo "ANTHROPIC_API_KEY=your_key_here" > .env
```

---

## Usage

```bash
# Ingest
python3 main.py ingest UBER
python3 main.py ingest UBER LYFT DASH
python3 main.py ingest UBER 10-K 3
python3 main.py list

# Query (terminal)
python3 main.py ask UBER "What are the main risk factors?"
python3 main.py compare UBER LYFT "autonomous vehicles"
python3 main.py delta UBER "autonomous vehicles" 2024 2026

# Browser UI
python3 server.py                                    # Terminal 1
cloudflared tunnel --url http://localhost:8000       # Terminal 2
open frontend.html                                   # Paste tunnel URL → Connect
```

---

## Delta Analysis

The delta query tracks any topic present in filing text:

| Query type | Example |
|---|---|
| Financial metrics | Revenue growth, operating margin, capex |
| Strategic priorities | Areas of investment, geographic expansion |
| Competitive positioning | How they describe competitive threats |
| Risk posture | Tone and severity around a given risk category |
| New topics | Issues that appear in one year but not another |

**Example — UBER autonomous vehicles 2024 → 2026:**
- Aurora investment: $629M unrealised gain → $802M unrealised loss (-$1.4B swing)
- Commercialisation language: "will launch" (2024) → no mention (2026)
- Threat framing shift: Waymo/Tesla (2024) → AI assistants and LLMs (2026)

---

## Roadmap

### Phase 1 ✅ Complete
- Natural language Q&A, multi-ticker bulk ingest, cross-company compare, FastAPI server + browser UI

### Phase 2 ← In Progress
- [x] Year-over-year delta queries
- [x] Section-aware chunk labeling
- [x] Delta tab in browser UI
- [ ] Hybrid search (BM25 + vector)
- [ ] 10-Q support

### Phase 3
- [ ] Watchlist with auto-ingest
- [ ] Company snapshot cards
- [ ] Additional data source connectors

---

## Tech Stack

Python 3.14 · sentence-transformers · numpy · FastAPI · BeautifulSoup · Claude Haiku · SEC EDGAR · Cloudflare Tunnel
