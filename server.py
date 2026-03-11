"""
server.py — FastAPI server wrapping the SEC RAG system

Run with:
    cd ~/sec-rag
    source venv/bin/activate
    python3 server.py

Server runs at: http://localhost:8000
API docs at:    http://localhost:8000/docs
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading embedding model and vector store...")
    from sentence_transformers import SentenceTransformer
    from vectorstore import VectorStore

    app_state["model"] = SentenceTransformer("all-MiniLM-L6-v2")
    app_state["store"] = VectorStore()
    print(f"Ready. {app_state['store'].count()} chunks in store.")
    yield
    app_state.clear()
    print("Server shut down cleanly.")


app = FastAPI(
    title="SEC Market Intelligence API",
    description="Query SEC filings (10-K/10-Q) using natural language",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_ngrok_header(request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["ngrok-skip-browser-warning"] = "true"
    return response


class IngestRequest(BaseModel):
    tickers: list[str]
    form_type: str = "10-K"
    count: int = 1

class AskRequest(BaseModel):
    question: str
    ticker: str | None = None

class CompareRequest(BaseModel):
    ticker_a: str
    ticker_b: str
    topic: str

class DeltaRequest(BaseModel):
    ticker: str
    topic: str
    year_a: int
    year_b: int

class FilingInfo(BaseModel):
    ticker: str
    company: str
    form: str
    date: str
    chunks: int

class ListResponse(BaseModel):
    total_filings: int
    total_chunks: int
    filings: list[FilingInfo]

class AnswerResponse(BaseModel):
    answer: str
    chunks_retrieved: int
    sources: list[dict]


@app.get("/robots.txt", response_class=PlainTextResponse)
def robots():
    return "User-agent: *\nAllow: /"

app.mount("/static", StaticFiles(directory="."), name="static")


@app.get("/")
def health_check():
    store = app_state.get("store")
    return {
        "status": "running",
        "chunks_in_store": store.count() if store else 0,
        "message": "SEC Market Intelligence API is live"
    }


@app.get("/list", response_model=ListResponse)
def list_filings():
    store = app_state["store"]

    if not store.documents:
        return ListResponse(total_filings=0, total_chunks=0, filings=[])

    seen = {}
    for doc in store.documents:
        meta = doc["metadata"]
        acc = meta.get("accession", "unknown")
        if acc not in seen:
            seen[acc] = {
                "ticker": meta.get("ticker", "?"),
                "company": meta.get("company", "?"),
                "form": meta.get("form", "?"),
                "date": meta.get("date", "?"),
                "chunks": 0,
            }
        seen[acc]["chunks"] += 1

    filings = [FilingInfo(**info) for info in seen.values()]
    filings.sort(key=lambda x: x.ticker)

    return ListResponse(total_filings=len(filings), total_chunks=store.count(), filings=filings)


@app.post("/ingest")
def ingest(req: IngestRequest):
    from ingest import ingest_filing

    results = []
    for ticker in req.tickers:
        ticker = ticker.upper()
        try:
            ingest_filing(ticker, req.form_type, req.count)
            store = app_state["store"]
            store._load()
            results.append({"ticker": ticker, "status": "success"})
        except Exception as e:
            results.append({"ticker": ticker, "status": "error", "detail": str(e)})

    app_state["store"] = __import__("vectorstore").VectorStore()

    return {"results": results, "total_chunks_now": app_state["store"].count()}


@app.post("/ask", response_model=AnswerResponse)
def ask_question(req: AskRequest):
    import anthropic as anthropic_sdk
    from query import build_context

    model = app_state["model"]
    store = app_state["store"]

    query_embedding = model.encode([req.question])[0].tolist()

    chunks = store.query(
        query_embedding,
        n_results=6,
        ticker=req.ticker.upper() if req.ticker else None
    )

    if not chunks:
        raise HTTPException(status_code=404, detail=f"No data found for ticker '{req.ticker}'. Run POST /ingest first.")

    context = build_context(chunks)

    client = anthropic_sdk.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        system="""You are a financial analyst assistant specializing in SEC filings.
Answer questions based ONLY on the provided excerpts from 10-K and 10-Q filings.
Be specific - cite exact numbers, dates, and quotes when available.
If the context does not contain enough information to answer, say so clearly.
Format your response in clear markdown.""",
        messages=[{"role": "user", "content": f"Context from SEC filings:\n\n{context}\n\n---\n\nQuestion: {req.question}"}]
    )

    return AnswerResponse(
        answer=response.content[0].text,
        chunks_retrieved=len(chunks),
        sources=[c["metadata"] for c in chunks],
    )


@app.post("/delta")
def delta_query(req: DeltaRequest):
    from query import delta as run_delta

    result = run_delta(req.ticker.upper(), req.topic, req.year_a, req.year_b)
    return {
        "ticker": req.ticker.upper(),
        "topic": req.topic,
        "year_a": req.year_a,
        "year_b": req.year_b,
        "answer": result,
    }


@app.post("/compare")
def compare_tickers(req: CompareRequest):
    import anthropic as anthropic_sdk
    from query import build_context

    model = app_state["model"]
    store = app_state["store"]

    query_embedding = model.encode([req.topic])[0].tolist()

    ticker_a = req.ticker_a.upper()
    ticker_b = req.ticker_b.upper()

    chunks_a = store.query(query_embedding, n_results=5, ticker=ticker_a)
    chunks_b = store.query(query_embedding, n_results=5, ticker=ticker_b)

    if not chunks_a:
        raise HTTPException(status_code=404, detail=f"No data for {ticker_a}. Run POST /ingest first.")
    if not chunks_b:
        raise HTTPException(status_code=404, detail=f"No data for {ticker_b}. Run POST /ingest first.")

    context_a = build_context(chunks_a)
    context_b = build_context(chunks_b)

    client = anthropic_sdk.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=2048,
        system="""You are a financial analyst comparing two companies using their SEC filings.
Be specific — cite exact numbers, dates, and direct language from the filings.
Highlight meaningful differences, not just surface-level observations.
Format your response in clear markdown.""",
        messages=[{"role": "user", "content": f"""Compare {ticker_a} and {ticker_b} on: "{req.topic}"

--- {ticker_a} FILING EXCERPTS ---
{context_a}

--- {ticker_b} FILING EXCERPTS ---
{context_b}

---

Provide:
1. **{ticker_a}** — what their filing says about "{req.topic}"
2. **{ticker_b}** — what their filing says about "{req.topic}"
3. **Key Differences** — the most important contrasts between the two"""}]
    )

    return {
        "ticker_a": ticker_a,
        "ticker_b": ticker_b,
        "topic": req.topic,
        "answer": response.content[0].text,
        "sources_a": [c["metadata"] for c in chunks_a],
        "sources_b": [c["metadata"] for c in chunks_b],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
