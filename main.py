import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient

load_dotenv()

app = FastAPI(
    title="ZAIO RAG Assistant API",
    description="RAG Q&A API over the ZAIO Student Handbook and ZAIO Website",
    version="2.0.0"
)

CHROMA_PATH = "vector_db"
COLLECTION_NAME = "handbook_collection"

HF_MODEL = os.getenv("HF_MODEL", "openai/gpt-oss-120b")

# Strict exact string required by assignment specs
NOT_FOUND_MESSAGE = "I could not find that information in the available knowledge base."

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
    """Combines unique sources ('Student Handbook - Page X' or website URLs) from chunks."""
    sources = []
    for chunk in retrieved_chunks:
        src = chunk.get("source", "Unknown Source")
        if src not in sources:
            sources.append(src)
    return ", ".join(sources) if sources else "N/A"


def synthesize_answer(question: str, retrieved_chunks: list) -> str:
    """Calls HuggingFace Inference Client to generate a grounded answer."""
    context = "\n\n".join(
        f"[Source: {chunk['source']}]\n{chunk['content']}" for chunk in retrieved_chunks
    )

    system_prompt = (
        "You are the ZAIO Assistant. Answer the user's question using ONLY the provided "
        "context excerpts below from the Student Handbook and ZAIO website. Do not use outside knowledge. "
        f"If the context does not contain enough information to answer, reply with exactly: "
        f'"{NOT_FOUND_MESSAGE}" Keep answers concise and conversational.'
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
    return {"message": "ZAIO RAG Assistant API is live! Navigate to /docs for Swagger UI."}


@app.post("/ask")
def ask_handbook(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    docs_and_scores = get_vector_store().similarity_search_with_relevance_scores(
        query=request.question, k=request.top_k
    )
    results = [doc for doc, score in docs_and_scores if score >= SCORE_THRESHOLD]

    # Return required refusal string if no relevant context clears similarity threshold
    if not results:
        return {"answer": NOT_FOUND_MESSAGE, "source": "N/A"}

    retrieved_chunks = [
        {
            "content": doc.page_content,
            "source": doc.metadata.get("source", "Unknown Source")
        }
        for doc in results
    ]

    try:
        answer = synthesize_answer(request.question, retrieved_chunks)
    except Exception as e:
        return {
            "answer": f"Sorry, I couldn't generate an answer right now ({e}).",
            "source": format_source(retrieved_chunks)
        }

    # Verify if LLM returned exact refusal text
    source = "N/A" if answer.strip() == NOT_FOUND_MESSAGE else format_source(retrieved_chunks)
    return {"answer": answer, "source": source}