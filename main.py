"""
main.py — CLI entrypoint for SEC RAG

Usage:
  python main.py ingest UBER                      # Ingest Uber's latest 10-K
  python main.py ingest UBER LYFT DASH            # Bulk ingest multiple tickers
  python main.py ingest UBER 10-K 3               # Ingest last 3 annual filings
  python main.py ingest UBER 10-Q 4               # Ingest last 4 quarterly filings

  python main.py list                             # Show all ingested filings
  python main.py query                            # Interactive Q&A across all filings
  python main.py query UBER                       # Interactive Q&A filtered to Uber
  python main.py ask UBER "What are risk factors?" # Single question, one ticker
  python main.py compare UBER LYFT "autonomous vehicles"  # Compare two companies
  python main.py delta UBER "revenue" 2024 2025           # YoY delta for one company
"""

import sys
from rich.console import Console

console = Console()


def main():
    if len(sys.argv) < 2:
        console.print(__doc__)
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "list":
        from vectorstore import VectorStore
        store = VectorStore()
        store.list_filings()

    elif command == "ingest":
        if len(sys.argv) < 3:
            console.print("[red]Provide at least one ticker: python main.py ingest UBER[/red]")
            sys.exit(1)

        from ingest import ingest_filing

        args = sys.argv[2:]

        def looks_like_ticker(s):
            return s.replace("-", "").isalpha() and s.upper() not in ("K", "Q")

        if len(args) > 1 and all(looks_like_ticker(a) for a in args):
            tickers = [a.upper() for a in args]
            console.print(f"[bold cyan]Bulk ingesting: {', '.join(tickers)}[/bold cyan]")
            for ticker in tickers:
                console.rule(f"[bold]{ticker}[/bold]")
                ingest_filing(ticker, "10-K", 1)
        else:
            ticker = args[0].upper()
            form = args[1] if len(args) > 1 else "10-K"
            count = int(args[2]) if len(args) > 2 else 1
            ingest_filing(ticker, form, count)

    elif command == "query":
        from query import interactive
        ticker = sys.argv[2].upper() if len(sys.argv) > 2 else None
        interactive(ticker)

    elif command == "ask":
        from query import ask
        from rich.markdown import Markdown
        from rich.panel import Panel

        ticker = sys.argv[2].upper() if len(sys.argv) > 2 else None
        question = sys.argv[3] if len(sys.argv) > 3 else None
        if not question:
            console.print("[red]Usage: python main.py ask UBER 'your question'[/red]")
            sys.exit(1)
        answer = ask(question, ticker=ticker, verbose=True)
        console.print(Panel(Markdown(answer), title="Answer", border_style="green"))

    elif command == "compare":
        from query import compare
        from rich.markdown import Markdown
        from rich.panel import Panel

        if len(sys.argv) < 5:
            console.print("[red]Usage: python main.py compare UBER LYFT 'topic or question'[/red]")
            sys.exit(1)

        ticker_a = sys.argv[2].upper()
        ticker_b = sys.argv[3].upper()
        question = sys.argv[4]
        answer = compare(ticker_a, ticker_b, question)
        console.print(Panel(Markdown(answer), title=f"[bold]{ticker_a} vs {ticker_b}[/bold]", border_style="cyan"))

    elif command == "delta":
        from query import delta
        from rich.markdown import Markdown
        from rich.panel import Panel

        if len(sys.argv) < 6:
            console.print("[red]Usage: python main.py delta UBER 'topic' 2024 2025[/red]")
            sys.exit(1)

        ticker = sys.argv[2].upper()
        topic = sys.argv[3]
        year_a = sys.argv[4]
        year_b = sys.argv[5]
        answer = delta(ticker, topic, year_a, year_b)
        console.print(Panel(Markdown(answer), title=f"[bold]{ticker}: {year_a} → {year_b} | {topic}[/bold]", border_style="yellow"))

    else:
        console.print(f"[red]Unknown command: {command}[/red]")
        console.print(__doc__)


if __name__ == "__main__":
    main()
