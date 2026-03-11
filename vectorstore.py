import os
import json
import numpy as np


class VectorStore:
    """Simple file-based vector store. No dependencies beyond numpy."""

    def __init__(self, path="./data/vectorstore"):
        os.makedirs(path, exist_ok=True)
        self.path = path
        self.docs_file = os.path.join(path, "documents.jsonl")
        self.vecs_file = os.path.join(path, "embeddings.npy")
        self.documents = []
        self.embeddings = None
        self._load()

    def _load(self):
        if os.path.exists(self.docs_file):
            with open(self.docs_file, "r") as f:
                self.documents = [json.loads(line) for line in f if line.strip()]

        if os.path.exists(self.vecs_file) and self.documents:
            self.embeddings = np.load(self.vecs_file)

    def _save(self):
        with open(self.docs_file, "w") as f:
            for doc in self.documents:
                f.write(json.dumps(doc) + "\n")
        if self.embeddings is not None:
            np.save(self.vecs_file, self.embeddings)

    def count(self):
        return len(self.documents)

    def has_accession(self, accession):
        return any(d["metadata"].get("accession") == accession for d in self.documents)

    def add(self, texts, embeddings, metadatas):
        new_vecs = np.array(embeddings, dtype=np.float32)
        for text, meta in zip(texts, metadatas):
            self.documents.append({"text": text, "metadata": meta})

        if self.embeddings is None:
            self.embeddings = new_vecs
        else:
            self.embeddings = np.vstack([self.embeddings, new_vecs])

        self._save()

    def list_filings(self):
        from rich.console import Console
        from rich.table import Table

        console = Console()

        if not self.documents:
            console.print("[yellow]No filings ingested yet. Run: python main.py ingest UBER[/yellow]")
            return

        seen = {}
        for doc in self.documents:
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

        table = Table(title=f"Ingested Filings ({len(seen)} total, {len(self.documents)} chunks)", show_lines=True)
        table.add_column("Ticker", style="bold cyan", width=8)
        table.add_column("Company", width=30)
        table.add_column("Form", width=8)
        table.add_column("Filed", width=12)
        table.add_column("Chunks", justify="right", width=8)

        for acc, info in sorted(seen.items(), key=lambda x: x[1]["ticker"]):
            table.add_row(info["ticker"], info["company"], info["form"], info["date"], str(info["chunks"]))

        console.print(table)

    def query(self, query_embedding, n_results=6, ticker=None):
        if not self.documents or self.embeddings is None:
            return []

        q = np.array(query_embedding, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-10)

        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        normed = self.embeddings / (norms + 1e-10)
        scores = normed @ q

        if ticker:
            mask = np.array([d["metadata"].get("ticker") == ticker.upper() for d in self.documents])
            scores = np.where(mask, scores, -1)

        top_indices = np.argsort(scores)[::-1][:n_results]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    "text": self.documents[idx]["text"],
                    "metadata": self.documents[idx]["metadata"],
                    "score": float(scores[idx])
                })

        return results
