import os
import json
import fitz  # PyMuPDF
from ebooklib import epub
from bs4 import BeautifulSoup

# ========== Parameters ==========
CHUNK_SIZE = 500  # Max words per chunk
INPUT_DIR = "data"  # Folder with EPUB + PDF files
OUTPUT_DIR = "processed_chunks"  # Where we'll save JSONL chunks

# Make sure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== Helper: Chunking ==========
# Improved chunking that avoids skipping short docs
def chunk_text(text, max_words=CHUNK_SIZE):
    """
    Splits the input text into smaller chunks.
    If the whole text is short, returns it as a single chunk.
    """
    words = text.split()
    if len(words) <= max_words:
        return [" ".join(words)] if words else []
    
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        chunks.append(chunk)
    return chunks


# ========== Helper: Process PDF ==========
def extract_text_from_pdf(file_path):
    """
    Uses PyMuPDF to extract text from a PDF file.
    Returns one string with all page text concatenated.
    """
    doc = fitz.open(file_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

# ========== Helper: Process EPUB ==========
from ebooklib.epub import EpubHtml

def extract_text_from_epub(file_path):
    """
    Extracts readable text from an EPUB file using ebooklib and BeautifulSoup.
    Returns one string with all content concatenated.
    """
    book = epub.read_epub(file_path)
    text = ""
    for item in book.get_items():
        # Only process HTML-like content
        if isinstance(item, EpubHtml):
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            cleaned = soup.get_text(separator=' ', strip=True)
            if cleaned:
                text += cleaned + "\n"
    return text



# ========== Main: Process All Files ==========
def process_all_files(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        name, ext = os.path.splitext(filename)

        if ext.lower() == ".pdf":
            print(f"📄 Processing PDF: {filename}")
            raw_text = extract_text_from_pdf(file_path)
            chunks = chunk_text(raw_text)
            output_file = os.path.join(output_dir, f"pdf_{name}.jsonl")

        elif ext.lower() == ".epub":
            print(f"📘 Processing EPUB: {filename}")
            raw_text = extract_text_from_epub(file_path)
            chunks = chunk_text(raw_text)
            output_file = os.path.join(output_dir, f"epub_{name}.jsonl")

        else:
            print(f"⚠️ Skipping unsupported file: {filename}")
            continue

        # Save chunks to JSONL
        with open(output_file, "w", encoding="utf-8") as f:
            for i, chunk in enumerate(chunks):
                json.dump({"id": f"{name}_{i}", "text": chunk}, f)
                f.write("\n")

        print(f"✅ Saved {len(chunks)} chunks to {output_file}\n")

# ========== Run It ==========
if __name__ == "__main__":
    process_all_files(INPUT_DIR, OUTPUT_DIR)
