# CityCare Hospital Information Bot (RAG)

Chatbot that answers questions on departments, doctors, visiting hours, admission, billing and facilities from hospital documents (fictional CityCare Hospital, in `doc/`).

## Architecture
```
doc/*.md -> chunk by "## " section -> MiniLM embeddings -> FAISS (cosine) -> top-3 chunks
         -> LLM answer grounded in context (or extractive fallback) -> Streamlit chat UI
```
- **Chunking:** section-level, with document title + heading prepended so each chunk is self-describing.
- **Retrieval:** `all-MiniLM-L6-v2` + `faiss.IndexFlatIP` on normalized vectors (= cosine similarity).
- **Guardrails:** similarity threshold (`MIN_SCORE`) refuses out-of-scope questions; the LLM prompt restricts answers to context and forbids medical advice.
- **Generation:** optional. With `GEMINI_API_KEY` set, Gemini writes the answer; otherwise the best section is returned verbatim.

## Run
```
pip install -r requirements.txt
streamlit run app.py
```
The app asks for your Gemini API key in the sidebar (free at https://aistudio.google.com/apikey). It is kept only in your browser session. For local use you can instead put `GEMINI_API_KEY=...` in a `.env` file (git-ignored).
Add your own `.md` files (with `## ` sections) to `doc/` to change the knowledge base.

## Learnings / limitations
- Section chunking beat fixed-size chunking for this small, well-structured corpus.
- Short queries like "deposit for private room" score low (0.27) - close to the threshold; hybrid BM25 + dense search or a reranker would help.
- Extractive mode returns whole sections; LLM mode gives concise answers.
