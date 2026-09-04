import os
import re
import shutil
import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

PDF_PATH = os.path.join("data", "handbook.pdf")
VECTOR_DB_DIR = "vector_db"
COLLECTION_NAME = "handbook_collection"

# URLs to crawl from ZAIO website
WEBSITE_URLS = [
    "https://www.zaio.io",
    "https://www.zaio.io/apply-now",
    "https://www.zaio.io/fullstack-ai-engineer-bootcamp",
    "https://www.zaio.io/datascience-bootcamp"
]

def strip_page_number_artifact(text: str) -> str:
    """
    Strips leading page-number footers from extracted text streams.
    Handles both space-separated ("16 We understand...") and 
    glued-together ("16We understand...") extraction artifacts.
    """
    return re.sub(r'^\s*\d{1,4}\s*(?=[A-Za-z])', '', text)

def normalize_extracted_text(text: str) -> str:
    """Clean text artifacts, extra spaces, and spacing issues."""
    # 1. Strip leading page numbers first so glued numbers (e.g. "13F inal") don't mess up character runs
    text = strip_page_number_artifact(text)

    # 2. Collapse single-character spaced runs (e.g., "F i n a l" -> "Final")
    def collapse_run(match: re.Match) -> str:
        return re.sub(r' ', '', match.group(0))

    text = re.sub(r'\b(?:[A-Za-z0-9] ){1,}[A-Za-z0-9]\b', collapse_run, text)

    # 3. Clean up punctuation spacing and extra white space
    text = re.sub(r'\s+([.,%:;!?])', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text
    """Clean text artifacts, extra spaces, and spacing issues."""
    def collapse_run(match: re.Match) -> str:
        return re.sub(r' ', '', match.group(0))

    text = re.sub(r'\b(?:[A-Za-z0-9] ){1,}[A-Za-z0-9]\b', collapse_run, text)
    text = strip_page_number_artifact(text)
    text = re.sub(r'\s+([.,%:;!?])', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def scrape_website_content(urls: list[str]) -> list[Document]:
    """Crawl, extract, and clean text content from ZAIO website pages."""
    web_documents = []
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    for url in urls:
        print(f"Crawling website URL: {url}...")
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                print(f"Warning: Failed to fetch {url} (Status: {response.status_code})")
                continue

            soup = BeautifulSoup(response.text, "html.parser")

            # Remove navigation, headers, footers, scripts, and styles
            for element in soup(["header", "footer", "nav", "script", "style", "noscript"]):
                element.decompose()

            # Extract main text content
            clean_text = soup.get_text(separator=" ")
            clean_text = normalize_extracted_text(clean_text)

            if clean_text:
                # Store metadata with explicit URL as required by Part 1
                doc = Document(
                    page_content=clean_text,
                    metadata={"source": url}
                )
                web_documents.append(doc)
        except Exception as e:
            print(f"Error scraping {url}: {e}")

    return web_documents

def run_ingestion():
    # 1. Process PDF Document
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"Source PDF not found at path: '{PDF_PATH}'.")

    print(f"Loading PDF document from: {PDF_PATH}...")
    loader = PyPDFLoader(PDF_PATH)
    pdf_docs = loader.load()

    # Clean PDF content & format handbook source metadata
    for doc in pdf_docs:
        doc.page_content = normalize_extracted_text(doc.page_content)
        page_num = doc.metadata.get("page", 0) + 1  # 1-indexed page number
        doc.metadata = {"source": f"Student Handbook - Page {page_num}"}

    # 2. Process Website Pages
    web_docs = scrape_website_content(WEBSITE_URLS)

    # 3. Combine & Chunk Content
    all_docs = pdf_docs + web_docs
    print(f"Loaded {len(pdf_docs)} PDF pages and {len(web_docs)} web pages.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=150,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(all_docs)
    print(f"Created {len(chunks)} total text chunks.")

    # 4. Rebuild Persistent Chroma Vector DB
    if os.path.exists(VECTOR_DB_DIR):
        print(f"Clearing existing Vector DB directory ('{VECTOR_DB_DIR}')...")
        shutil.rmtree(VECTOR_DB_DIR)

    print("Initializing HuggingFace Embeddings (sentence-transformers/all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print(f"Ingesting chunks into ChromaDB at '{VECTOR_DB_DIR}'...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIR,
        collection_name=COLLECTION_NAME,
        collection_metadata={"hnsw:space": "cosine"}
    )

    print("Ingestion complete! Multi-source vector database created.")


if __name__ == "__main__":
    run_ingestion()