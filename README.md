**ZAIO Handbook Assistant - RAG System**

_A free, local-first RAG pipeline for the ZAIO Bootcamp handbook_

# 1\. Overview

A Retrieval-Augmented Generation (RAG) API that answers student questions about the ZAIO Full-Stack AI Engineer Bootcamp 2026 Handbook.

The system loads the handbook PDF, cleans up PDF-extraction text artifacts, splits it into chunks, embeds those chunks, and stores them in a local vector database. At query time it retrieves the most relevant chunks for a question and asks a free LLM to generate a grounded answer from them - falling back to a clear "not found" message when nothing relevant exists.

## Tech Stack

| **Component**    | **Choice**                                                           | **Cost**                       |
| ---------------- | -------------------------------------------------------------------- | ------------------------------ |
| API framework    | FastAPI                                                              | Free                           |
| PDF loading      | langchain-community (PyPDFLoader)                                    | Free                           |
| Text splitting   | langchain-text-splitters (RecursiveCharacterTextSplitter)            | Free                           |
| Embeddings       | HuggingFace sentence-transformers/all-MiniLM-L6-v2 (runs locally)    | Free                           |
| Vector database  | ChromaDB (langchain-chroma, persisted locally, cosine distance)      | Free                           |
| LLM (generation) | HuggingFace hosted Inference API (openai/gpt-oss-120b, configurable) | Free (token required, no card) |

_Everything in this stack is free - embeddings run locally on your machine, and HuggingFace's free hosted Inference API is used for answer generation instead of a paid provider._

# 2\. Project Structure
```
handbook-assistant/  
data/  
handbook.pdf # Source ZAIO Bootcamp 2026 handbook  
vector_db/ # Persistent ChromaDB storage (from ingest.py)  
ingest.py # Load, clean, chunk, embed, store  
main.py # Retrieval + generation + /ask endpoint  
tests/  
test_ingest.py # Unit tests for text cleaning  
test_api.py # Unit tests for the /ask endpoint  
requirements.txt  
.env.example  
README.md
```

# 3\. Setup

\# Clone and enter the project  
```
cd handbook-assistant
```
<br/>\# Create and activate a virtual environment  
```
python -m venv venv  
source venv/bin/activate # macOS/Linux  
.\\venv\\Scripts\\Activate.ps1 # Windows PowerShell
```
<br/>\# Install dependencies  
```
pip install -r requirements.txt
```

## Get a free HuggingFace token

- Sign up at huggingface.co (no credit card required).
- Create a token at huggingface.co/settings/tokens - read access is enough.
- Copy .env.example to .env and paste the token in.
  
```
HF_TOKEN=your_free_huggingface_token_here  
HF_MODEL=openai/gpt-oss-120b
```

## Add the handbook

Place your handbook.pdf in the data/ folder.

# 4\. Processing the Handbook

Run the ingestion pipeline once (and again any time the PDF changes):

python ingest.py

This:

- Loads the handbook PDF (PyPDFLoader)
- Cleans PDF-extraction artifacts - character-spaced text (e.g. "T h i s" → "This") and a page-number footer glued onto the first word - via normalize_extracted_text()
- Splits the cleaned text into overlapping chunks (RecursiveCharacterTextSplitter, 700 chars, 150 overlap)
- Generates embeddings locally (all-MiniLM-L6-v2, 384-dim, free, no API key)
- Stores the embeddings in a persistent ChromaDB collection using cosine distance (vector_db/)

# 5\. Retrieval, Generation, and the API

Start the API:
```
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Swagger UI: <http://127.0.0.1:8000/docs>

## POST /ask

When a question comes in, the API:

- Embeds the question
- Searches ChromaDB for the most relevant chunks, filtering out anything below a relevance-score threshold
- Sends the surviving chunks as context to a free HuggingFace-hosted LLM, which generates the answer
- Returns the answer plus the handbook page(s) it came from - or the not-found message if nothing relevant was retrieved

**Request:**
```
{  
"question": "What is the attendance requirement?"  
}
```

**Response:**
```
{  
"answer": "...",  
"source": "Page 12"  
}
```

top_k (optional, default 3) controls how many candidate chunks are retrieved before relevance filtering:
```
{ "question": "What is the assignment weighting?", "top_k": 5 }
```

If no relevant chunks clear the relevance threshold:

```
{  
"answer": "I'm sorry, but I couldn't find any  
information regarding that in the  
student handbook.",  
"source": "N/A"  
}
```

## Automated unit tests

```
pytest tests/ -v
```

Covers:

- Text-cleaning helpers (strip_page_number_artifact, normalize_extracted_text) - including regression tests against the real handbook PDF's actual extraction structure
- /ask returning a grounded answer with the correct source page(s)
- /ask filtering out low-relevance chunks so unrelated pages don't leak into the answer or source
- /ask returning the not-found message when nothing is retrieved or nothing clears the relevance threshold
- /ask rejecting empty questions (400)
- /ask handling malformed, missing-field, or wrong-type requests gracefully (422, structured JSON)
- /ask degrading gracefully (200 with an explanatory message) if the LLM call fails

_Tests mock the vector store and the HuggingFace client, so they run in under two seconds with no ingested database, live token, or network access required._

# 6\. n8n Integration (Planned)

The API is ready to be wired into an n8n workflow:

- Accepts JSON - POST /ask with { "question": "...", "top_k": 3 }
- Returns JSON - { "answer": "...", "source": "..." }
- Handles invalid requests gracefully - empty questions return 400; missing/malformed fields return a structured 422 from FastAPI/Pydantic validation rather than crashing

n8n HTTP Request node configuration (planned next step):

- Method: POST
- URL: http://&lt;your-server-ip&gt;:8000/ask
- Headers: Content-Type: application/json
- Body: { "question": "={{ \$json.incoming_chat_message }}" }
- Downstream: pipe {{ \$json.answer }} into Slack, WhatsApp, or Discord.

_n8n integration is the next planned step for this project._

# 7\. Design Notes - Retrieval Relevance Filtering

Rather than always returning exactly top_k chunks regardless of quality, /ask uses Chroma's similarity_search_with_relevance_scores() and drops any chunk below SCORE_THRESHOLD before it reaches either the LLM's context or the source field. This prevents weakly-related pages (e.g. the cover page) from leaking into an otherwise unrelated answer.

The threshold was calibrated empirically against this collection: genuinely relevant chunks scored ~0.40-0.43, while an irrelevant cover-page chunk scored ~0.24 for the same query. SCORE_THRESHOLD = 0.3 was chosen to sit in that gap. This requires the vector store to use cosine distance (set via collection_metadata={"hnsw:space": "cosine"} in both ingest.py and main.py) - Chroma's default L2 distance is not bounded 0-1 and produces uncalibrated, sometimes negative relevance scores.

_If re-used with a different embedding model or document set, re-check this threshold: print the scores for a handful of real questions and confirm relevant vs. irrelevant chunks still separate cleanly around the chosen cutoff._
