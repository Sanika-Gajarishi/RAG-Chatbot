import os
import pickle
import fitz  # PyMuPDF
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --------------------------------------------------
# Configuration
# --------------------------------------------------

DOCUMENTS_PATH = "data/documents"
INDEX_PATH = "data/faiss_index"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --------------------------------------------------
# Load Embedding Model
# --------------------------------------------------

print("Loading embedding model...")
model = SentenceTransformer(EMBEDDING_MODEL)

# --------------------------------------------------
# Text Splitter
# --------------------------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

# --------------------------------------------------
# Read PDFs and Create Chunks
# --------------------------------------------------

chunks = []

print("\nReading documents...\n")

for filename in os.listdir(DOCUMENTS_PATH):

    if not filename.lower().endswith(".pdf"):
        continue

    pdf_path = os.path.join(DOCUMENTS_PATH, filename)

    print(f"Processing: {filename}")

    pdf = fitz.open(pdf_path)

    for page_number in range(len(pdf)):

        page = pdf.load_page(page_number)

        text = page.get_text().strip()

        if not text:
            continue

        page_chunks = splitter.split_text(text)

        for chunk in page_chunks:

            chunks.append({
                "document": filename,
                "page": page_number + 1,
                "text": chunk
            })

    pdf.close()

print(f"\nTotal chunks created: {len(chunks)}")

# --------------------------------------------------
# Generate Embeddings
# --------------------------------------------------

print("\nGenerating embeddings...\n")

texts = [chunk["text"] for chunk in chunks]

embeddings = model.encode(
    texts,
    convert_to_numpy=True,
    show_progress_bar=True
)

print(f"Embedding Shape: {embeddings.shape}")

# --------------------------------------------------
# Build FAISS Index
# --------------------------------------------------

print("\nCreating FAISS Index...\n")

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings).astype("float32"))

# --------------------------------------------------
# Save FAISS Index
# --------------------------------------------------

os.makedirs(INDEX_PATH, exist_ok=True)

faiss.write_index(
    index,
    os.path.join(INDEX_PATH, "faiss.index")
)

print("FAISS index saved successfully.")

# --------------------------------------------------
# Save Metadata
# --------------------------------------------------

metadata = []

for idx, chunk in enumerate(chunks):

    metadata.append({
        "id": idx,
        "chunk_id": idx,
        "document": chunk["document"],
        "page": chunk["page"],
        "text": chunk["text"]
    })

metadata_path = os.path.join(INDEX_PATH, "metadata.pkl")

with open(metadata_path, "wb") as f:
    pickle.dump(metadata, f)

print("Metadata saved successfully.")

# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\n==============================")
print("Ingestion Completed Successfully")
print("==============================")

print(f"Documents Indexed : {len(set([c['document'] for c in chunks]))}")
print(f"Total Chunks      : {len(chunks)}")
print(f"Vector Dimension  : {dimension}")

print("\nFiles Generated:")
print(f"✓ {os.path.join(INDEX_PATH, 'faiss.index')}")
print(f"✓ {metadata_path}")

print("\nDocuments found in metadata:")

documents = set()

for item in metadata:
    documents.add(item["document"])

print(documents)