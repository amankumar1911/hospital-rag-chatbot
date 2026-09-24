"""RAG pipeline for the CityCare Hospital information bot.

Flow: load docs -> chunk by section -> embed (sentence-transformers) -> Chroma vector DB
-> retrieve top-k for a question -> answer with an LLM (if configured) or extractively.
"""
import os
import re
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv

load_dotenv()  # reads GEMINI_API_KEY from .env if present

DOC_DIR = Path(__file__).parent / "doc"
DB_DIR = Path(__file__).parent / "chroma_db"  # persisted on disk, survives restarts
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.6-flash")
MIN_SCORE = 0.25  # below this cosine similarity, treat the question as out of scope


@dataclass
class Chunk:
    text: str
    source: str
    heading: str


def load_chunks(doc_dir: Path = DOC_DIR) -> list[Chunk]:
    """Split each markdown file on '## ' headings so a chunk is one coherent topic."""
    chunks = []
    for path in sorted(doc_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title = re.match(r"# (.+)", text)
        title = title.group(1) if title else path.stem
        for section in re.split(r"\n(?=## )", text)[1:]:
            heading = section.splitlines()[0].lstrip("# ").strip()
            # Prefix with title/heading so the embedding carries the context.
            chunks.append(Chunk(f"{title} – {heading}\n{section}", path.name, heading))
    return chunks


class HospitalRAG:
    def __init__(self, doc_dir: Path = DOC_DIR, db_dir: Path = DB_DIR):
        self.chunks = load_chunks(doc_dir)
        embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
        # PersistentClient writes the index to disk under db_dir, so it survives restarts
        # and doesn't need re-embedding every time the app starts (unlike the old FAISS setup).
        client = chromadb.PersistentClient(path=str(db_dir))
        self.collection = client.get_or_create_collection(
            "hospital_docs", embedding_function=embed_fn, metadata={"hnsw:space": "cosine"})
        # Re-sync if the docs changed since the last run (cheap check: chunk count).
        if self.collection.count() != len(self.chunks):
            client.delete_collection("hospital_docs")
            self.collection = client.get_or_create_collection(
                "hospital_docs", embedding_function=embed_fn, metadata={"hnsw:space": "cosine"})
            self.collection.add(
                ids=[str(i) for i in range(len(self.chunks))],
                documents=[c.text for c in self.chunks],
                metadatas=[{"source": c.source, "heading": c.heading} for c in self.chunks])

    @staticmethod
    def make_client(api_key: str | None = None):
        """Build a Gemini client from the given key (or env). Returns None if no key."""
        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            return None
        from google import genai
        return genai.Client(api_key=key)

    def retrieve(self, question: str, k: int = 3) -> list[tuple[Chunk, float]]:
        res = self.collection.query(query_texts=[question], n_results=k)
        hits = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            similarity = 1 - dist  # cosine distance -> cosine similarity
            hits.append((Chunk(doc, meta["source"], meta["heading"]), similarity))
        return hits

    def answer(self, question: str, history: list[dict] | None = None, api_key: str | None = None) -> dict:
        hits = self.retrieve(question)
        if not hits or hits[0][1] < MIN_SCORE:
            return {"answer": "I couldn't find that in the hospital documents. "
                              "Please call the main reception at +91-22-5550-1000.",
                    "sources": []}
        sources = [{"source": c.source, "heading": c.heading, "score": round(s, 2)} for c, s in hits]
        llm = self.make_client(api_key)
        if llm is None:
            # Extractive fallback: return the best-matching section verbatim.
            body = hits[0][0].text.split("\n", 1)[1]
            return {"answer": body.strip(), "sources": sources}

        context = "\n\n---\n\n".join(c.text for c, _ in hits)
        system = ("You are the CityCare Hospital information assistant. Answer ONLY from the "
                  "context below. If the answer is not there, say you don't know and suggest "
                  "calling reception. Be concise. Never give medical advice or diagnoses.\n\n"
                  f"Context:\n{context}")
        from google.genai import types
        turns = (history or [])[-6:] + [{"role": "user", "content": question}]
        contents = [types.Content(role="user" if t["role"] == "user" else "model",
                                  parts=[types.Part(text=t["content"])]) for t in turns]
        try:
            resp = llm.models.generate_content(
                model=LLM_MODEL, contents=contents,
                config=types.GenerateContentConfig(system_instruction=system, max_output_tokens=500))
        except Exception as e:  # bad key, quota, model unavailable...
            return {"answer": f"⚠️ Gemini request failed: {getattr(e, 'message', None) or e}", "sources": sources}
        return {"answer": resp.text, "sources": sources}
