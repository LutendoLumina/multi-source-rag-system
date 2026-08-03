**RAG System - Full-Stack AI Engineer Handbook Assistant**



\_**\_**\_**\_**\_**\_**\_**\_**\_**\_**\_**\_**\_**\_**\_**\_**\_**\_**\_**\_**\_**\_**\_**\_**\_**\_**\___

**1\. Executive Overview**

This repository implements an enterprise-grade Retrieval-Augmented Generation (RAG) system engineered to parse, clean, index, and answer queries regarding the ZAIO Full-Stack AI Engineer Bootcamp 2026 Handbook.

The system addresses severe text-parsing artifacts (such as character-level spacing caused by PDF stream extraction) through automated regex-based normalization before generating 384-dimensional dense vector embeddings. Search queries are processed through high-precision vector similarity matching using ChromaDB and FastAPI.

**2\. System Architecture & Tech Stack**

- **Core Language:** Python 3.10+
- **API Framework:** FastAPI (Async OpenAPI UI / Swagger support)
- **Document Processing:** langchain-community (PyPDFLoader), Regex Normalization Pipeline
- **Vector Database:** langchain-chroma (ChromaDB Persistent Engine)
- **Embeddings:** langchain-huggingface (HuggingFace sentence-transformers/all-MiniLM-L6-v2)
- **Data Validation:** Pydantic v2

**3\. Project Structure**

zaio-handbook-rag/  
├── data/  
│ └── handbook.pdf # Source ZAIO Bootcamp 2026 Handbook PDF  
├── vector_db/ # Persistent ChromaDB vector database storage  
├── ingest.py # Automated parsing, regex normalization & chunking pipeline  
├── main.py # FastAPI server exposing similarity search & synthesized RAG endpoints  
├── requirements.txt # Explicit dependency locking file  
└── README.md # Complete system setup and verification documentation

**4\. Environment Setup & Execution Instructions**

**Step 1: Environment Setup**

\# Clone repository and navigate to project root  
cd zaio-handbook-rag  
<br/>\# Create virtual environment  
python -m venv .venv  
<br/>\# Activate virtual environment (Windows PowerShell)  
.\\.venv\\Scripts\\Activate.ps1  
\# Activate virtual environment (macOS/Linux)  
source .venv/bin/activate  
<br/>\# Install locked dependencies  
pip install -r requirements.txt

**Step 2: Execute Vector Ingestion Pipeline**

Run ingest.py to clear stale vector data, clean source PDF text using character-level normalization, generate embeddings, and build the persistent vector collection:

python ingest.py

**Step 3: Launch FastAPI Web Service**

Start the API server using Uvicorn:

uvicorn main:app --reload --host 127.0.0.1 --port 8000

Access interactive Swagger documentation UI at: <http://127.0.0.1:8000/docs>

**5\. API Specification**

**POST /query**

Accepts a JSON payload containing the student query and desired return depth (top_k).

**Request Payload Example:**

{  
"question": "What is the grade breakdown for the final project?",  
"top_k": 3  
}

**Response Payload Example:**

{  
"question": "What is the grade breakdown for the final project?",  
"answer": "Final Project ( React + Node ) ( 35 % ) : The final project serves as a comprehensive evaluation...",  
"results_count": 3,  
"retrieved_chunks": \[  
{  
"chunk_id": 1,  
"content": "Final Project ( React + Node ) ( 35 % ) : The final project serves as a comprehensive evaluation...",  
"page": 13  
}  
\]  
}

**6\. Comprehensive Testing & Evaluation (10 Meaningful Questions)**

The following evaluation log documents testing across 10 distinct queries covering diverse handbook topics, verifying retrieval precision, source page metadata mapping, and graceful fallback handling.

| **#** | **Question**                                       | **Status**            | **Source Page** | **Retrieved Answer / Output**                                                                                                                        |
| ----- | -------------------------------------------------- | --------------------- | --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1     | What is the grade breakdown for the final project? | **PASSED**            | Page 13         | Final Project (React + Node) counts for 35% of the total bootcamp mark, serving as a comprehensive evaluation of full-stack application development. |
| 2     | When are tutor support hours?                      | **PASSED**            | Page 10, 13     | Tutors are available on standby to assist students studying asynchronously or who have doubts outside of scheduled class times.                      |
| 3     | When are live classes held?                        | **PASSED**            | Page 10         | Live classes run Tuesdays (9 AM - 11 AM) and Thursdays (6 PM - 8 PM) for the first 12 weeks, then reduce to Tuesdays (9 AM - 11 AM).                 |
| 4     | What are the laptop system requirements?           | **PASSED**            | Page 4          | Dual-core Intel i5 / AMD 3000+ / Apple M1 CPU or higher, with at least 4 GB or 8 GB of RAM and Windows (PowerShell) or Mac (Terminal).               |
| 5     | What is the weighting for assignments?             | **PASSED**            | Page 13         | Assignments account for 25% of the overall final grade and gauge independent application of hands-on coding and problem-solving concepts.            |
| 6     | What is the grade weight of coding challenges?     | **PASSED**            | Page 13         | Coding challenges make up 25% of the total grade, assessing participants' algorithmic efficiency, critical thinking, and coding proficiency.         |
| 7     | How much are MCQs worth in the final grade?        | **PASSED**            | Page 13         | MCQs account for 15% of the final mark, evaluating theoretical knowledge and core computer science/AI conceptual understanding.                      |
| 8     | What topics are covered in the bootcamp?           | **PASSED**            | Page 3, 5       | The Full-Stack AI Engineer Bootcamp covers full-stack web development (React, Node) integrated with AI application engineering and automation.       |
| 9     | What happens if I miss a live class?               | **PASSED**            | Page 10         | All live sessions are recorded for students unable to attend live, with standby tutor support available on the LMS platform.                         |
| 10    | What is the policy for refunding tuition fees?     | **FALLBACK VERIFIED** | N/A (0 Chunks)  | I'm sorry, but I couldn't find any information regarding that in the student handbook.                                                               |

**7\. n8n Automation Workflow Integration**

This API is fully optimized for external workflow automation platforms such as n8n:

- **HTTP Request Node Method:** POST
- **URL:** http://&lt;your-server-ip&gt;:8000/query
- **Header Configuration:** Content-Type: application/json
- **Body Expression:** { "question": "={{ \$json.incoming_chat_message }}", "top_k": 3 }
- **Workflow Output Routing:** Pass {{ \$json.answer }} directly into Slack, WhatsApp Webhooks, or Discord bots.