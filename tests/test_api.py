import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
import main

client = TestClient(main.app)


class FakeDoc:
    """Mimics a langchain Document returned by similarity_search."""
    def __init__(self, content, page):
        self.page_content = content
        self.metadata = {"page": page}


class FakeVectorStore:
    def __init__(self, docs):
        self.docs = docs

    def similarity_search(self, query, k=3):
        return self.docs[:k]


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeCompletion:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


class FakeChatCompletions:
    def __init__(self, response_text):
        self.response_text = response_text

    def create(self, **kwargs):
        return FakeCompletion(self.response_text)


class FakeChat:
    def __init__(self, response_text):
        self.completions = FakeChatCompletions(response_text)


class FakeGroqClient:
    """Mimics the Groq SDK's client.chat.completions.create(...) call shape."""
    def __init__(self, response_text):
        self.chat = FakeChat(response_text)


def test_ask_returns_answer_and_source(monkeypatch):
    fake_docs = [FakeDoc("Assignments are worth 25% of the final grade.", 13)]
    monkeypatch.setattr(main, "get_vector_store", lambda: FakeVectorStore(fake_docs))
    monkeypatch.setattr(
        main, "get_groq_client",
        lambda: FakeGroqClient("Assignments count for 25% of your final grade.")
    )

    response = client.post("/ask", json={"question": "What is the assignment weighting?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Assignments count for 25% of your final grade."
    assert body["source"] == "Page 13"


def test_ask_returns_multiple_pages_when_chunks_span_pages(monkeypatch):
    fake_docs = [
        FakeDoc("Tutors are available Tuesdays.", 20),
        FakeDoc("Support also runs Thursdays.", 20),
        FakeDoc("Live classes recap the week's material.", 10),
    ]
    monkeypatch.setattr(main, "get_vector_store", lambda: FakeVectorStore(fake_docs))
    monkeypatch.setattr(main, "get_groq_client", lambda: FakeGroqClient("Tutors support Tuesdays and Thursdays."))

    response = client.post("/ask", json={"question": "When is tutor support?", "top_k": 3})

    assert response.status_code == 200
    assert response.json()["source"] == "Page 20, Page 10"


def test_ask_returns_not_found_when_no_chunks_retrieved(monkeypatch):
    monkeypatch.setattr(main, "get_vector_store", lambda: FakeVectorStore([]))

    response = client.post("/ask", json={"question": "What is the meaning of life?"})

    assert response.status_code == 200
    body = response.json()
    assert "couldn't find any information" in body["answer"]
    assert body["source"] == "N/A"


def test_ask_rejects_empty_question(monkeypatch):
    monkeypatch.setattr(main, "get_vector_store", lambda: FakeVectorStore([FakeDoc("x", 1)]))
    response = client.post("/ask", json={"question": "   "})
    assert response.status_code == 400


def test_ask_handles_missing_required_field_gracefully():
    # No "question" key at all -> FastAPI/Pydantic validation should
    # return a structured 422 JSON error, not crash the server.
    response = client.post("/ask", json={"top_k": 3})
    assert response.status_code == 422
    assert response.json()["detail"]


def test_ask_falls_back_gracefully_if_llm_call_fails(monkeypatch):
    fake_docs = [FakeDoc("Some handbook content.", 5)]
    monkeypatch.setattr(main, "get_vector_store", lambda: FakeVectorStore(fake_docs))

    class BrokenGroqClient:
        @property
        def chat(self):
            raise RuntimeError("Groq API unavailable")

    monkeypatch.setattr(main, "get_groq_client", lambda: BrokenGroqClient())

    response = client.post("/ask", json={"question": "Anything?"})

    assert response.status_code == 200
    body = response.json()
    assert "couldn't generate an answer" in body["answer"]
    assert body["source"] == "Page 5"
