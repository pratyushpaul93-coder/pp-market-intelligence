import os
import anthropic
from sentence_transformers import SentenceTransformer
from vectorstore import VectorStore
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from dotenv import load_dotenv

load_dotenv()
console = Console()

_model = None
_store = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def get_store():
    global _store
    if _store is None:
        _store = VectorStore()
    return _store


def retrieve(query, ticker=None, n_results=6):
    model = get_model()
    store = get_store()
    query_embedding = model.encode([query])[0].tolist()
    return store.query(query_embedding, n_results=n_results, ticker=ticker)


def build_context(chunks):
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk["metadata"]
        parts.append(
            f"[Source {i}: {meta.get('ticker')} {meta.get('form')} "
            f"filed {meta.get('date')} | Section: {meta.get('section', 'Unknown')}]\n"
            f"{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)


def ask(question, ticker=None, verbose=False):
    chunks = retrieve(question, ticker=ticker)

    if verbose:
        console.print(f"\n[dim]Retrieved {len(chunks)} chunks:[/dim]")
        for c in chunks:
            console.print(f"  [dim]- {c['metadata'].get('section')} (score: {c['score']:.3f})[/dim]")

    context = build_context(chunks)
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    system = """You are a financial analyst assistant specializing in SEC filings.
Answer questions based ONLY on the provided excerpts from 10-K and 10-Q filings.
Be specific - cite exact numbers, dates, and quotes when available.
If the context does not contain enough information to answer, say so clearly.
Format your response in clear markdown."""

    prompt = f"""Context from SEC filings:\n\n{context}\n\n---\n\nQuestion: {question}"""

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text


def compare(ticker_a, ticker_b, topic, n_results=5):
    """Retrieve context for both tickers and generate a side-by-side comparison."""
    model = get_model()
    store = get_store()

    query_embedding = model.encode([topic])[0].tolist()

    chunks_a = store.query(query_embedding, n_results=n_results, ticker=ticker_a)
    chunks_b = store.query(query_embedding, n_results=n_results, ticker=ticker_b)

    console.print(f"\n[dim]Retrieved {len(chunks_a)} chunks for {ticker_a}, {len(chunks_b)} for {ticker_b}[/dim]")

    if not chunks_a:
        return f"No data found for {ticker_a}. Run: python main.py ingest {ticker_a}"
    if not chunks_b:
        return f"No data found for {ticker_b}. Run: python main.py ingest {ticker_b}"

    context_a = build_context(chunks_a)
    context_b = build_context(chunks_b)

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    system = """You are a financial analyst comparing two companies using their SEC filings.
Be specific — cite exact numbers, dates, and direct language from the filings.
Highlight meaningful differences, not just surface-level observations.
Format your response in clear markdown with a section for each company followed by a direct comparison."""

    prompt = f"""Compare {ticker_a} and {ticker_b} on the topic: "{topic}"

--- {ticker_a} FILING EXCERPTS ---
{context_a}

--- {ticker_b} FILING EXCERPTS ---
{context_b}

---

Provide:
1. **{ticker_a}** — what their filing says about "{topic}"
2. **{ticker_b}** — what their filing says about "{topic}"
3. **Key Differences** — the most important contrasts between the two companies"""

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=2048,
        system=system,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text


def delta(ticker, topic, year_a, year_b, n_results=5):
    """Compare what a company said about a topic across two different filing years."""
    model = get_model()
    store = get_store()

    query_embedding = model.encode([topic])[0].tolist()

    all_chunks = store.query(query_embedding, n_results=50, ticker=ticker)

    def year_of(chunk):
        return chunk["metadata"].get("date", "")[:4]

    chunks_a = [c for c in all_chunks if year_of(c) == str(year_a)][:n_results]
    chunks_b = [c for c in all_chunks if year_of(c) == str(year_b)][:n_results]

    if not chunks_a:
        return f"No {ticker} filing data found for {year_a}. Run: python main.py ingest {ticker} 10-K 3"
    if not chunks_b:
        return f"No {ticker} filing data found for {year_b}. Run: python main.py ingest {ticker} 10-K 3"

    context_a = build_context(chunks_a)
    context_b = build_context(chunks_b)

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    system = """You are a financial analyst tracking year-over-year changes in SEC filings.
Focus on what CHANGED between years — not just what each year says.
Be specific: cite exact numbers, percentages, and direct quotes.
Highlight shifts in tone, strategy, risk posture, and financial metrics.
Format your response in clear markdown."""

    prompt = f"""Analyse how {ticker}'s position on "{topic}" changed from {year_a} to {year_b}.

--- {ticker} {year_a} FILING EXCERPTS ---
{context_a}

--- {ticker} {year_b} FILING EXCERPTS ---
{context_b}

---

Provide:
1. **{year_a} Position** — what the {year_a} filing says about "{topic}"
2. **{year_b} Position** — what the {year_b} filing says about "{topic}"
3. **What Changed** — the most significant shifts between the two years (numbers, tone, strategy)
4. **Signal** — what this change suggests for investors"""

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=2048,
        system=system,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text


def interactive(ticker=None):
    store = get_store()
    total = store.count()

    console.print(Panel(
        f"[bold cyan]SEC Market Intelligence[/bold cyan]\n"
        f"[dim]{total} chunks in database[/dim]\n"
        f"{'Filtered to: ' + ticker.upper() if ticker else 'Querying all tickers'}\n\n"
        f"Type your question or 'quit' to exit.",
        title="SEC RAG"
    ))

    while True:
        try:
            question = console.input("\n[bold yellow]Question:[/bold yellow] ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if question.lower() in ("quit", "exit", "q"):
            break

        if not question:
            continue

        console.print("[dim]Retrieving and generating...[/dim]")
        answer = ask(question, ticker=ticker, verbose=True)
        console.print(Panel(Markdown(answer), title="Answer", border_style="green"))


if __name__ == "__main__":
    import sys
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else None
    interactive(ticker)
