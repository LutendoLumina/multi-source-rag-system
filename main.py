import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient

load_dotenv()

app = FastAPI(
    title="ZAIO Full Stack AI Engineer Handbook Assistant API",
    description="RAG-based Q&A API over the ZAIO Full-Stack AI Engineer Bootcamp Handbook",
    version="2.0.0"
)

CHROMA_PATH = "vector_db"
COLLECTION_NAME = "handbook_collection"

HF_MODEL = os.getenv("HF_MODEL", "openai/gpt-oss-120b")

NOT_FOUND_MESSAGE = "I'm sorry, but I couldn't find any information regarding that in the student handbook."

SCORE_THRESHOLD = 0.3

_vector_store = None
_hf_client = None


def get_vector_store() -> Chroma:
    global _vector_store
    if _vector_store is None:
        embedding_function = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        _vector_store = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embedding_function,
            collection_name=COLLECTION_NAME,
            collection_metadata={"hnsw:space": "cosine"}
        )
    return _vector_store


def get_hf_client() -> InferenceClient:
    global _hf_client
    if _hf_client is None:
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        _hf_client = InferenceClient(token=token)
    return _hf_client


class AskRequest(BaseModel):
    question: str
    top_k: int = 3


def format_source(retrieved_chunks: list) -> str:
    """Builds a 'Page X' or 'Page X, Y' string from the unique pages used."""
    pages = []
    for chunk in retrieved_chunks:
        page = chunk["page"]
        if page not in pages:
            pages.append(page)
    return ", ".join(f"Page {page}" for page in pages) if pages else "N/A"


def synthesize_answer(question: str, retrieved_chunks: list) -> str:
    """Calls the free HuggingFace Inference API to generate a grounded
    answer from the retrieved chunks, rather than returning raw chunk text."""
    context = "\n\n".join(
        f"[Page {chunk['page']}] {chunk['content']}" for chunk in retrieved_chunks
    )

    system_prompt = (
        "You are the ZAIO Full-Stack AI Engineer Bootcamp Handbook Assistant. "
        "Answer the student's question using ONLY the information in the "
        "provided context excerpts below. Do not use outside knowledge. If "
        "the context does not contain enough information to answer, reply "
        "with exactly: \"I'm sorry, but I couldn't find any information "
        "regarding that in the student handbook.\" Keep answers concise "
        "and conversational."
    )

    completion = get_hf_client().chat.completions.create(
        model=HF_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ],
        max_tokens=500,
    )
    return completion.choices[0].message.content.strip()


@app.get("/")
def read_root():
    return {"message": "Handbook Assistant API is live! Navigate to /docs for Swagger UI."}


@app.post("/ask")
def ask_handbook(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    docs_and_scores = get_vector_store().similarity_search_with_relevance_scores(
        query=request.question, k=request.top_k
    )
    results = [doc for doc, score in docs_and_scores if score >= SCORE_THRESHOLD]

    if not results:
        return {"answer": NOT_FOUND_MESSAGE, "source": "N/A"}

    retrieved_chunks = [
        {"content": doc.page_content, "page": doc.metadata.get("page", "Unknown")}
        for doc in results
    ]

    try:
        answer = synthesize_answer(request.question, retrieved_chunks)
    except Exception as e:
        return {
            "answer": f"Sorry, I couldn't generate an answer right now ({e}).",
            "source": format_source(retrieved_chunks)
        }

    source = "N/A" if answer.strip() == NOT_FOUND_MESSAGE else format_source(retrieved_chunks)
    return {"answer": answer, "source": source}