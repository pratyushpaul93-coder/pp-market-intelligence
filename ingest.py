import os
import re
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from vectorstore import VectorStore
from rich.console import Console
from rich.progress import track

console = Console()

HEADERS = {
    "User-Agent": "SEC-RAG-Project contact@example.com",
    "Accept-Encoding": "gzip, deflate"
}

EDGAR_BASE = "https://data.sec.gov"


def get_cik(ticker):
    url = f"https://www.sec.gov/cgi-bin/browse-edgar?company=&CIK={ticker}&type=10-K&dateb=&owner=include&count=1&search_text=&action=getcompany&output=atom"
    resp = requests.get(url, headers=HEADERS)
    match = re.search(r'CIK=(\d+)', resp.text)
    if not match:
        url2 = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type=10-K&dateb=&owner=include&count=1&output=atom"
        resp2 = requests.get(url2, headers=HEADERS)
        match = re.search(r'/cgi-bin/browse-edgar\?action=getcompany&CIK=(\d+)', resp2.text)
    if not match:
        raise ValueError(f"Could not find CIK for ticker: {ticker}")
    return match.group(1).zfill(10)


def get_filings(cik, form_type="10-K", count=3):
    url = f"{EDGAR_BASE}/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=HEADERS)
    data = resp.json()

    filings = data["filings"]["recent"]
    results = []

    for i, form in enumerate(filings["form"]):
        if form == form_type and len(results) < count:
            results.append({
                "form": form,
                "date": filings["filingDate"][i],
                "accession": filings["accessionNumber"][i].replace("-", ""),
                "cik": cik,
                "company": data.get("name", "Unknown"),
            })

    return results


def get_filing_document_url(cik, accession):
    formatted = f"{accession[:10]}-{accession[10:12]}-{accession[12:]}"
    index_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{formatted}-index.htm"

    resp = requests.get(index_url, headers=HEADERS)
    soup = BeautifulSoup(resp.text, "lxml")

    for row in soup.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) >= 4:
            doc_type = cells[3].get_text(strip=True)
            if doc_type in ("10-K", "10-Q", "10-K/A"):
                link = cells[2].find("a")
                if link:
                    return "https://www.sec.gov" + link["href"]

    for link in soup.find_all("a", href=True):
        href = link["href"]
        if href.endswith(".htm") and "index" not in href.lower():
            return "https://www.sec.gov" + href

    return None


def fetch_and_parse(url):
    if "/ix?doc=" in url:
        url = "https://www.sec.gov" + url.split("/ix?doc=")[1]

    resp = requests.get(url, headers=HEADERS, timeout=30)
    soup = BeautifulSoup(resp.content, "lxml")

    for tag in soup.find_all(["script", "style", "table"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    lines = [l for l in lines if l]
    return "\n".join(lines)


SECTION_NAMES = {
    "1":  "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity",
    "2":  "Properties",
    "3":  "Legal Proceedings",
    "4":  "Mine Safety Disclosures",
    "5":  "Market for Registrant Equity",
    "6":  "Selected Financial Data",
    "7":  "MD&A",
    "7A": "Quantitative Disclosures About Market Risk",
    "8":  "Financial Statements",
    "9":  "Changes in Disagreements with Accountants",
    "9A": "Controls and Procedures",
    "9B": "Other Information",
    "10": "Directors and Executive Officers",
    "11": "Executive Compensation",
    "12": "Security Ownership",
    "13": "Certain Relationships",
    "14": "Principal Accountant Fees",
    "15": "Exhibits",
}


def normalise_section(raw_header):
    match = re.match(r'Item\s+(\d+[A-C]?)\.?', raw_header, re.IGNORECASE)
    if match:
        key = match.group(1).upper()
        name = SECTION_NAMES.get(key, raw_header.strip())
        return f"Item {key} — {name}"
    return raw_header.strip()


def chunk_by_section(text, metadata, chunk_size=1000):
    chunks = []

    section_pattern = re.compile(
        r'(Item\s+\d+[A-C]?\.\s+[A-Z][^\n]{0,60})',
        re.MULTILINE
    )

    sections = section_pattern.split(text)

    if len(sections) > 3:
        current_section = "Cover / Preamble"
        for part in sections:
            if section_pattern.match(part):
                current_section = normalise_section(part)
            else:
                words = part.split()
                for i in range(0, len(words), chunk_size):
                    chunk_words = words[i:i + chunk_size]
                    if len(chunk_words) < 50:
                        continue
                    chunks.append({
                        "text": " ".join(chunk_words),
                        "section": current_section,
                        **metadata
                    })
    else:
        words = text.split()
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i + chunk_size]
            if len(chunk_words) < 50:
                continue
            chunks.append({
                "text": " ".join(chunk_words),
                "section": "Unknown",
                **metadata
            })

    return chunks


def ingest_filing(ticker, form_type="10-K", count=1):
    console.print(f"\n[bold cyan]Looking up {ticker} {form_type} filings...[/bold cyan]")

    console.print("[yellow]Loading embedding model (first run downloads ~90MB)...[/yellow]")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    store = VectorStore()
    console.print(f"[green]Vector store loaded - {store.count()} chunks existing[/green]")

    cik = get_cik(ticker)
    console.print(f"[green]CIK: {cik}[/green]")

    filings = get_filings(cik, form_type, count)
    if not filings:
        console.print(f"[red]No {form_type} filings found for {ticker}[/red]")
        return

    for filing in filings:
        console.print(f"\n[bold]Processing {filing['form']} filed {filing['date']} - {filing['company']}[/bold]")

        if store.has_accession(filing["accession"]):
            console.print("[yellow]Already ingested, skipping.[/yellow]")
            continue

        doc_url = get_filing_document_url(filing["cik"], filing["accession"])
        if not doc_url:
            console.print("[red]Could not find document URL, skipping.[/red]")
            continue

        console.print(f"[dim]URL: {doc_url}[/dim]")
        console.print("Parsing HTML...")
        text = fetch_and_parse(doc_url)
        console.print(f"[green]Extracted {len(text):,} characters[/green]")

        metadata = {
            "ticker": ticker.upper(),
            "company": filing["company"],
            "form": filing["form"],
            "date": filing["date"],
            "accession": filing["accession"],
        }
        chunks = chunk_by_section(text, metadata)
        console.print(f"[green]Created {len(chunks)} chunks[/green]")

        batch_size = 50
        for i in track(range(0, len(chunks), batch_size), description="Embedding..."):
            batch = chunks[i:i + batch_size]
            texts = [c["text"] for c in batch]
            embeddings = model.encode(texts, show_progress_bar=False).tolist()
            metadatas = [{k: v for k, v in c.items() if k != "text"} for c in batch]
            store.add(texts, embeddings, metadatas)

        console.print(f"[bold green]Done! {len(chunks)} chunks stored.[/bold green]")

    console.print(f"\n[bold green]Total chunks in store: {store.count()}[/bold green]")


if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    form = sys.argv[2] if len(sys.argv) > 2 else "10-K"
    ingest_filing(ticker, form)
