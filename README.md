# Multi-Source RAG System (Student Handbook & ZAIO Website)

A Python-native Retrieval-Augmented Generation (RAG) system built with **FastAPI**, **ChromaDB**, **LangChain**, and **HuggingFace Inference API**. This API ingests unstructured documentation from both a static PDF (Student Handbook) and crawled web content (ZAIO Website), enabling accurate contextual question answering via a local n8n workflow.

---

## Architecture Overview

```
[ PDF Document ] -----                       +---> [ ingest.py ] ---> [ ChromaDB Vector Store ]
[ Web Scraper ]  -----/         (Embeddings)               |
                                                           | (Similarity Search + Threshold)
                                                           v
[ n8n Workflow ] ---> POST /ask ---> [ FastAPI main.py ] --+
                                           |
                                           v
                             [ HuggingFace Inference API ]
                            (openai/gpt-oss-120b Model)
```

1. **Ingestion (`ingest.py`)**: Extracts text from `Student Handbook.pdf` using `PyPDF2`/`pdfplumber` and crawls pages from `https://www.zaio.io` using `requests` and `BeautifulSoup`. Text is normalized, chunked into overlapping segments, and stored in ChromaDB alongside metadata tags (`source`).
2. **Vector Retrieval & Filtering**: Uses `sentence-transformers/all-MiniLM-L6-v2` locally for fast embedding generation. Performs cosine similarity search with a strict relevance score threshold (**>= 0.3**).
3. **LLM Generation**: Connects to the HuggingFace Inference API (`openai/gpt-oss-120b`) to answer queries grounded exclusively in retrieved context.
4. **Refusal Logic**: If no chunks pass the `0.3` similarity threshold, the API immediately short-circuits and returns:
   > *"I could not find that information in the available knowledge base."*
5. **Orchestration**: Fully integrable with **n8n Community Edition** via HTTP POST requests.

---

## Folder Structure

```
.
├── data/
│   └── student_handbook.pdf    # Source PDF document
├── chroma_db/                  # Local persistent Chroma vector database
├── tests/
│   ├── test_ingest.py          # Unit tests for text cleaning and ingestion logic
│   └── test_api.py             # Mocked unit tests for FastAPI endpoints
├── .env                        # Local environment variables (git-ignored)
├── .env.example                # Example configuration template
├── ingest.py                   # Data ingestion and chunking script
├── main.py                     # FastAPI application endpoints and RAG pipeline
├── requirements.txt            # Python dependencies
└── README.md                   # System documentation
```

---

## Setup & Installation

### 1. Prerequisites
* Python 3.10+
* Free HuggingFace Account & Access Token ([Get HF Token](https://huggingface.co/settings/tokens))

### 2. Virtual Environment Setup
Clone the repository and create a virtual environment:

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows PowerShell)
. env\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the project root directory (refer to `.env.example`):

```env
HF_TOKEN=hf_your_actual_token_here
HF_MODEL=openai/gpt-oss-120b
SCORE_THRESHOLD=0.3
CHROMA_DB_DIR=./chroma_db
```

---

## Usage Guide

### Step 1: Run Knowledge Base Ingestion
Execute `ingest.py` to scrape the web content, process the handbook PDF, build embeddings, and store vector indexes in `chroma_db/`:

```powershell
python ingest.py
```

### Step 2: Start the FastAPI Server
Launch the development server via Uvicorn:

```powershell
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The interactive Swagger UI documentation will be accessible at:
👉 **`http://127.0.0.1:8000/docs`**

---

## API Endpoints

### `POST /ask`
Submits a user query for RAG processing.

#### Request Body
```json
{
  "question": "What courses does ZAIO offer?",
  "top_k": 3
}
```

#### Successful Response (`200 OK`)
```json
{
  "answer": "ZAIO’s course catalogue includes Full-Stack AI Engineer, Cloud & DevOps Engineer, Full-Stack Web Development, Data Science, Cybersecurity, and Digital Marketing bootcamps.",
  "source": "https://www.zaio.io, Student Handbook - Page 14"
}
```

#### Refusal Response (`200 OK` - Below Relevance Threshold)
```json
{
  "answer": "I could not find that information in the available knowledge base.",
  "source": "N/A"
}
```

#### Validation Error (`400 Bad Request` / `422 Unprocessable Entity`)
Returned when empty questions or malformed JSON payloads are supplied.

---

## Running Unit Tests

Unit tests use `pytest` and mock external calls (Vector Store and HuggingFace API) to allow instant offline testing:

```powershell
pytest tests/ -v
```

---

## n8n Integration

To integrate with **n8n Community Edition** locally:

1. Launch n8n locally (`npx n8n` or via Docker on port `5678`).
2. Add an **HTTP Request** node with the following configuration:
   * **Method**: `POST`
   * **URL**: `http://localhost:8000/ask` (or `http://host.docker.internal:8000/ask` if n8n is running in Docker)
   * **Send Body**: `ON`
   * **Body Content Type**: `JSON`
   * **JSON Body**:
     ```json
     {
       "question": "={{ $json.question }}"
     }
     ```
3. Execute the node to receive structured JSON outputs containing `answer` and `source`.