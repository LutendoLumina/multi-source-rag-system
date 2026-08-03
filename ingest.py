import os
import re
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

PDF_PATH = os.path.join("data", "handbook.pdf")
VECTOR_DB_DIR = "vector_db"
COLLECTION_NAME = "handbook_collection"


def strip_page_number_artifact(text: str) -> str:
    """
    PyPDFLoader sometimes extracts the printed page-number footer as part of
    the page's text stream, glued directly onto the first word of real
    content (e.g. "16We understand the importance..."). Strip a leading
    run of 1-4 digits that is immediately followed by a letter with no
    space in between, since real sentence content never starts that way.
    """
    return re.sub(r'^\s*\d{1,4}(?=[A-Za-z])', '', text)


def normalize_extracted_text(text: str) -> str:
    text = strip_page_number_artifact(text)

    # Some PDFs extract text with every character separated by a single
    # space (e.g. "T h i s   i s"). Find whole RUNS of 3+ single
    # alphanumeric characters each separated by exactly one space, and
    # only collapse spaces within that run. A run this long is virtually
    # never a sequence of real one-letter words, so this is safe against
    # false positives on genuine short words like "I" or "a".
    def collapse_run(match: re.Match) -> str:
        return re.sub(r'\s+', '', match.group(0))

    text = re.sub(r'\b(?:[A-Za-z0-9]\s){1,}[A-Za-z0-9]\b', collapse_run, text)

    # Remove stray whitespace before punctuation (e.g. "35 %" -> "35%",
    # "word ," -> "word,") left behind by extraction or the collapse step.
    text = re.sub(r'\s+([.,%:;!?])', r'\1', text)

    # Collapse any remaining runs of whitespace (including newlines) into
    # a single space, then trim the ends.
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def run_ingestion():
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"Source PDF not found at path: '{PDF_PATH}'. Please check the directory.")

    print(f"Loading PDF document from: {PDF_PATH}...")
    loader = PyPDFLoader(PDF_PATH)
    raw_documents = loader.load()

    print("Cleaning and normalizing extracted text...")
    for doc in raw_documents:
        doc.page_content = normalize_extracted_text(doc.page_content)

    print("Chunking document into text segments...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=150,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(raw_documents)
    print(f"Created {len(chunks)} text chunks across {len(raw_documents)} pages.")

    # Remove existing vector store directory to avoid mixing old embeddings
    if os.path.exists(VECTOR_DB_DIR):
        print(f"Clearing existing Vector DB directory ('{VECTOR_DB_DIR}')...")
        shutil.rmtree(VECTOR_DB_DIR)

    print("Initializing Hugging Face Embeddings (free, runs locally)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print(f"Ingesting chunks into ChromaDB at '{VECTOR_DB_DIR}'...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIR,
        collection_name=COLLECTION_NAME
    )

    print("Ingestion complete! Vector database successfully created.")


if __name__ == "__main__":
    run_ingestion()