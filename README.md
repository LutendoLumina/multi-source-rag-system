# ZAIO Handbook Assistant — RAG System

A Retrieval-Augmented Generation (RAG) API that answers student questions about the
**ZAIO Full-Stack AI Engineer Bootcamp 2026 Handbook**, built for the RAG System assignment.

## 1. Overview

The system loads the handbook PDF, cleans up PDF-extraction text artifacts, splits it into
chunks, embeds those chunks, and stores them in a local vector database. At query time it
retrieves the most relevant chunks for a question and asks a free LLM to generate a grounded
answer from them — falling back to a clear "not found" message when nothing relevant exists.

**Tech stack**

| Component | Choice | Cost |
|---|---|---|
| API framework | FastAPI | Free |
| PDF loading | `langchain-community` (`PyPDFLoader`) | Free |
| Text splitting | `langchain-text-splitters` | Free |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (runs locally) | Free |
| Vector database | ChromaDB (`langchain-chroma`, persisted locally) | Free |
| LLM (generation) | **Groq** (free tier — `llama-3.1-8b-instant`) | Free, no credit card |

Everything in this stack is free — embeddings run locally on your machine, and Groq's free
tier is used for answer generation instead of a paid API.

## 2. Project Structure

```
zaio-handbook-rag/
├── data/
│   └── handbook.pdf          # Source ZAIO Bootcamp 2026 handbook
├── vector_db/                 # Persistent ChromaDB storage (created by ingest.py)
├── ingest.py                  # Part 1: load, clean, chunk, embed, store
├── main.py                    # Parts 2 & 3: retrieval + generation + /ask endpoint
├── tests/
│   ├── test_ingest.py         # Unit tests for text cleaning
│   └── test_api.py            # Unit tests for the /ask endpoint
├── requirements.txt
├── .env.example
└── README.md
```

## 3. Setup

```bash
# Clone and enter the project
cd zaio-handbook-rag

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .\.venv\Scripts\Activate.ps1    # Windows PowerShell

# Install dependencies
pip install -r requirements.txt
```

### Get a free Groq API key

1. Go to [console.groq.com](https://console.groq.com) and sign up (no credit card required).
2. Create an API key at [console.groq.com/keys](https://console.groq.com/keys).
3. Copy `.env.example` to `.env` and paste your key in:

```bash
cp .env.example .env
```

```
GROQ_API_KEY=your_free_groq_api_key_here
```

### Add the handbook

Place your `handbook.pdf` in the `data/` folder.

## 4. Part 1 — Process the Handbook

Run the ingestion pipeline once (and again any time the PDF changes):

```bash
python ingest.py
```

This:
- ✅ Loads the handbook PDF (`PyPDFLoader`)
- ✅ Cleans PDF-extraction artifacts — stray footer page numbers glued onto text, and
  character-spaced text runs (e.g. `"T h i s"` → `"This"`) — via `normalize_extracted_text()`
- ✅ Splits the cleaned text into overlapping chunks (`RecursiveCharacterTextSplitter`)
- ✅ Generates embeddings locally (`all-MiniLM-L6-v2`, 384-dim, free)
- ✅ Stores the embeddings in a persistent ChromaDB collection (`vector_db/`)

## 5. Part 2 & 3 — Retrieval, Generation, and the API

Start the API:

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Swagger UI: http://127.0.0.1:8000/docs

### `POST /ask`

When a question comes in, the API:
1. Embeds the question
2. Searches ChromaDB for the most relevant chunks
3. Sends those chunks as context to a free Groq LLM, which generates the actual answer
4. Returns the answer plus the handbook page(s) it came from — or the not-found message if
   nothing relevant was retrieved

**Request:**
```json
{
  "question": "What is the attendance requirement?"
}
```

**Response:**
```json
{
  "answer": "...",
  "source": "Page 12"
}
```

`top_k` (optional, default `3`) controls how many chunks are retrieved:
```json
{ "question": "What is the assignment weighting?", "top_k": 5 }
```

If no relevant chunks are found:
```json
{
  "answer": "I'm sorry, but I couldn't find any information regarding that in the student handbook.",
  "source": "N/A"
}
```

## 6. Part 4 — Testing the Assistant

Manual evaluation log across 10 questions covering diverse handbook topics:

| # | Question | Source (Page) | Answer |
|---|---|---|---|
| 1 | What is the grade breakdown for the final project? | Page 13 | Final Project (React + Node) counts for 35% of the total bootcamp mark, evaluating full-stack development proficiency. |
| 2 | When are tutor support hours? | Page 20 | Tutor support runs Tuesdays 2–4pm and 6–8pm, and Thursdays 10am–12pm and 6–8pm, bookable via Calendly or the ZAIO dashboard. |
| 3 | When are live classes held? | Page 10 | Live classes run Tuesdays 9–11am and Thursdays 6–8pm for the first 12 weeks, then reduce to Tuesdays 9–11am only. |
| 4 | What are the laptop system requirements? | Page 4 | A dual-core Intel i5 / AMD 3000+ / Apple M1 or better, with 4–8GB RAM, an SSD, and a stable internet connection. |
| 5 | What is the weighting for assignments? | Page 13 | Assignments make up 25% of the final grade. |
| 6 | What is the grade weight of coding challenges? | Page 13 | Coding challenges make up 25% of the final grade. |
| 7 | How much are MCQs worth in the final grade? | Page 13 | MCQs account for 15% of the final mark. |
| 8 | What topics are covered in the bootcamp? | Page 3, Page 5 | The bootcamp covers full-stack web development (HTML/CSS/JS, React, Node) integrated with AI agent engineering (LangChain, n8n). |
| 9 | What happens if I miss a live class? | Page 10 | All live sessions are recorded, and tutors are available on standby for students who study asynchronously. |
| 10 | What is the policy for refunding tuition fees? | N/A | I'm sorry, but I couldn't find any information regarding that in the student handbook. |

> Note: exact page numbers depend on `PyPDFLoader`'s (zero-indexed) page metadata — verify
> against your own ingested PDF and adjust with a +1 offset if your printed page numbers are
> consistently one higher.

### Automated unit tests

```bash
pytest
```

Covers:
- Text-cleaning helpers (`strip_page_number_artifact`, `normalize_extracted_text`)
- `/ask` returning a grounded answer + correct source page(s)
- `/ask` returning the not-found message when nothing is retrieved
- `/ask` rejecting empty questions (400)
- `/ask` handling malformed/missing-field requests gracefully (422, structured JSON)
- `/ask` degrading gracefully (200 with an explanatory message) if the LLM call fails

Tests mock the vector store and Groq client, so they run without needing a real ingested
database or a live API key.

## 7. Part 5 — n8n Readiness

The API is ready to be wired into an n8n workflow:
- **Accepts JSON** — `POST /ask` with `{"question": "...", "top_k": 3}`
- **Returns JSON** — `{"answer": "...", "source": "..."}`
- **Handles invalid requests gracefully** — empty questions return `400`; missing/malformed
  fields return a structured `422` from FastAPI/Pydantic validation rather than crashing

**n8n HTTP Request node configuration** (to be wired up in the next practical):
- Method: `POST`
- URL: `http://<your-server-ip>:8000/ask`
- Headers: `Content-Type: application/json`
- Body: `{ "question": "={{ $json.incoming_chat_message }}" }`
- Downstream: pipe `{{ $json.answer }}` into Slack, WhatsApp, or Discord.

n8n integration itself is out of scope for this submission and will be completed next practical.
