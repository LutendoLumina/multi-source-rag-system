"""
Unit tests for main.py — Parts 2, 3, and 5 (Retrieval, API, n8n readiness).

The vector store and HuggingFace Inference client are mocked so these tests
run instantly without needing an ingested database, a live HF token, or
network access — useful both for local development and for grading.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
import main

client = TestClient(main.app)


# ---------------------------------------------------------------------------
# Test doubles mimicking the real langchain / huggingface_hub response shapes
# ---------------------------------------------------------------------------

class FakeDoc:
    """Mimics a langchain Document."""
    def __init__(self, content, source):
        self.page_content = content
        self.metadata = {"source": source}


class FakeVectorStore:
    """Mimics Chroma's similarity_search_with_relevance_scores."""
    def __init__(self, docs_and_scores):
        self.docs_and_scores = docs_and_scores

    def similarity_search_with_relevance_scores(self, query, k=3):
        return self.docs_and_scores[:k]


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeCompletion:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


class FakeCompletions:
    def __init__(self, response_text):
        self.response_text = response_text

    def create(self, **kwargs):
        return FakeCompletion(self.response_text)


class FakeChat:
    def __init__(self, response_text):
        self.completions = FakeCompletions(response_text)


class FakeHFClient:
    """Mimics huggingface_hub.InferenceClient's client.chat.completions.create(...) shape."""
    def __init__(self, response_text):
        self.chat = FakeChat(response_text)


# ---------------------------------------------------------------------------
# Part 2 & 3: retrieval + generation + response shape
# ---------------------------------------------------------------------------

def test_ask_returns_answer_and_source_for_handbook(monkeypatch):
    docs_and_scores = [(FakeDoc("Assignments are worth 25% of the final grade.", "Student Handbook - Page 13"), 0.42)]
    monkeypatch.setattr(main, "get_vector_store", lambda: FakeVectorStore(docs_and_scores))
    monkeypatch.setattr(
        main, "get_hf_client",
        lambda: FakeHFClient("Assignments count for 25% of your final grade.")
    )

    response = client.post("/ask", json={"question": "What is the assignment weighting?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Assignments count for 25% of your final grade."
    assert body["source"] == "Student Handbook - Page 13"


def test_ask_returns_answer_and_source_for_website(monkeypatch):
    docs_and_scores = [(FakeDoc("ZAIO offers Web Development and Data Science courses.", "https://www.zaio.io"), 0.50)]
    monkeypatch.setattr(main, "get_vector_store", lambda: FakeVectorStore(docs_and_scores))
    monkeypatch.setattr(
        main, "get_hf_client",
        lambda: FakeHFClient("ZAIO offers Web Development and Data Science bootcamps.")
    )

    response = client.post("/ask", json={"question": "What courses are offered?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "ZAIO offers Web Development and Data Science bootcamps."
    assert body["source"] == "https://www.zaio.io"


def test_ask_response_has_exactly_the_required_keys(monkeypatch):
    docs_and_scores = [(FakeDoc("Some content.", "Student Handbook - Page 1"), 0.5)]
    monkeypatch.setattr(main, "get_vector_store", lambda: FakeVectorStore(docs_and_scores))
    monkeypatch.setattr(main, "get_hf_client", lambda: FakeHFClient("An answer."))

    response = client.post("/ask", json={"question": "Anything?"})

    assert set(response.json().keys()) == {"answer", "source"}


def test_ask_filters_out_low_relevance_chunks(monkeypatch):
    docs_and_scores = [
        (FakeDoc("Hardware requirements: i5 CPU, 8GB RAM.", "Student Handbook - Page 4"), 0.43),
        (FakeDoc("ZAIO 2026 Full-Stack AI Engineer Bootcamp Handbook", "Student Handbook - Page 1"), 0.24),
    ]
    monkeypatch.setattr(main, "get_vector_store", lambda: FakeVectorStore(docs_and_scores))
    monkeypatch.setattr(main, "get_hf_client", lambda: FakeHFClient("You need at least an i5 CPU and 8GB RAM."))

    response = client.post("/ask", json={"question": "What are the hardware requirements?"})

    assert response.status_code == 200
    assert response.json()["source"] == "Student Handbook - Page 4"


def test_ask_returns_multiple_sources_when_chunks_span_sources(monkeypatch):
    docs_and_scores = [
        (FakeDoc("Tutors are available Tuesdays.", "Student Handbook - Page 20"), 0.5),
        (FakeDoc("Course information on web site.", "https://www.zaio.io"), 0.45),
    ]
    monkeypatch.setattr(main, "get_vector_store", lambda: FakeVectorStore(docs_and_scores))
    monkeypatch.setattr(main, "get_hf_client", lambda: FakeHFClient("Information from handbook and website."))

    response = client.post("/ask", json={"question": "Where can I get help?", "top_k": 2})

    assert response.status_code == 200
    assert response.json()["source"] == "Student Handbook - Page 20, https://www.zaio.io"


# ---------------------------------------------------------------------------
# Part 2: Respond with exact refusal string when unavailable
# ---------------------------------------------------------------------------

def test_ask_returns_not_found_when_no_chunks_retrieved(monkeypatch):
    monkeypatch.setattr(main, "get_vector_store", lambda: FakeVectorStore([]))

    response = client.post("/ask", json={"question": "What is the meaning of life?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == main.NOT_FOUND_MESSAGE
    assert body["source"] == "N/A"


def test_ask_returns_not_found_when_all_chunks_below_threshold(monkeypatch):
    docs_and_scores = [(FakeDoc("Unrelated cover page text.", "Student Handbook - Page 1"), 0.1)]
    monkeypatch.setattr(main, "get_vector_store", lambda: FakeVectorStore(docs_and_scores))

    response = client.post("/ask", json={"question": "What is the refund policy?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == main.NOT_FOUND_MESSAGE
    assert body["source"] == "N/A"


# ---------------------------------------------------------------------------
# Part 5: JSON parsing and error handling
# ---------------------------------------------------------------------------

def test_ask_rejects_empty_question(monkeypatch):
    monkeypatch.setattr(main, "get_vector_store", lambda: FakeVectorStore([(FakeDoc("x", "Student Handbook - Page 1"), 0.9)]))
    response = client.post("/ask", json={"question": "   "})
    assert response.status_code == 400
    assert "detail" in response.json()


def test_ask_handles_missing_required_field_gracefully():
    response = client.post("/ask", json={"top_k": 3})
    assert response.status_code == 422
    assert response.json()["detail"]


def test_ask_handles_wrong_type_gracefully():
    response = client.post("/ask", json={"question": 12345})
    assert response.status_code == 422


def test_ask_handles_malformed_json_body_gracefully():
    response = client.post(
        "/ask",
        content="{not valid json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


def test_ask_falls_back_gracefully_if_llm_call_fails(monkeypatch):
    docs_and_scores = [(FakeDoc("Some handbook content.", "Student Handbook - Page 5"), 0.5)]
    monkeypatch.setattr(main, "get_vector_store", lambda: FakeVectorStore(docs_and_scores))

    class BrokenHFClient:
        @property
        def chat(self):
            raise RuntimeError("HF Inference API unavailable")

    monkeypatch.setattr(main, "get_hf_client", lambda: BrokenHFClient())

    response = client.post("/ask", json={"question": "Anything?"})

    assert response.status_code == 200
    body = response.json()
    assert "couldn't generate an answer" in body["answer"]
    assert body["source"] == "Student Handbook - Page 5"


def test_root_endpoint_is_reachable():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()