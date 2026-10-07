# 📄 Document Q&A with Citations (RAG System)

A Retrieval-Augmented Generation (RAG) system built using **FastAPI**, **FAISS**, **Sentence Transformers**, **Gemini**, and **Streamlit**. The application answers questions from uploaded PDF documents, provides citations for every answer, compares two documents on a given topic, and supports multilingual queries.

---

## Features

- Document ingestion from PDF files
- Recursive text chunking
- Semantic embeddings using Sentence Transformers
- FAISS vector store for efficient similarity search
- Retrieval-Augmented Generation (RAG)
- `/ask` endpoint with citations
- `/contradict` endpoint for document comparison
- Multilingual query support (language detection + translation)
- Streamlit web interface
- Hallucination prevention using prompt constraints

---

## Project Architecture

```text
                PDF Documents
                      │
                      ▼
                Text Extraction
               (PyMuPDF / fitz)
                      │
                      ▼
          Recursive Character Chunking
                      │
                      ▼
      Sentence Transformer Embeddings
                      │
                      ▼
               FAISS Vector Store
                      │
                      ▼
             Similarity Retrieval
                      │
                      ▼
            Gemini 2.5 Flash LLM
                      │
                      ▼
      Answer with Supporting Citations
```

---

## Project Structure

```text
Rag-Assignment/
│
├── app/
│   ├── ingest.py
│   ├── rag.py
│   ├── translator.py
│   ├── contradiction.py
│   ├── main.py
│
├── data/
│   ├── documents/
│   └── faiss_index/
│
├── ui/
│   └── app.py
│
├── requirements.txt
├── README.md
└── .env
```

---

# Documents Used

The system indexes the following five PDF documents:

1. Chemistry.pdf
2. Foundations of GenAI & Tools.pdf
3. Gujarat Wind Order 2024.pdf
4. ICSICE.pdf
5. Maharashtra Renewable Energy and Energy Storage Policy.pdf

These documents were selected because they contain sufficient technical and policy content to evaluate retrieval, citations, multilingual querying, and document comparison.

---

# Chunking Strategy

The documents are processed page by page using **PyMuPDF**.

Each page is divided using LangChain's `RecursiveCharacterTextSplitter` with:

- Chunk Size: **800 characters**
- Chunk Overlap: **150 characters**

### Why this strategy?

Large chunks may exceed the model context window and reduce retrieval precision, while very small chunks often lose semantic meaning. An 800-character chunk with a 150-character overlap preserves contextual continuity while maintaining accurate semantic retrieval.

Each chunk stores:

- Document name
- Page number
- Chunk ID
- Chunk text

This metadata is later used to generate citations.

---

# Embedding Model

Sentence Transformer

```
sentence-transformers/all-MiniLM-L6-v2
```

Reason for selection:

- Lightweight
- Fast inference
- Good semantic retrieval quality
- Produces 384-dimensional embeddings
- Free to use

---

# Vector Store

FAISS (Facebook AI Similarity Search)

The generated embeddings are indexed in FAISS to perform efficient nearest-neighbor semantic search.

Artifacts generated:

```
data/faiss_index/
    faiss.index
    metadata.pkl
```

---

# Retrieval Pipeline

1. User submits a question.
2. Language is detected.
3. If necessary, the query is translated to English.
4. The query is embedded.
5. FAISS retrieves the most relevant chunks.
6. Retrieved chunks are provided as context to Gemini.
7. Gemini generates an answer using only the retrieved context.
8. The answer is translated back to the original language (if applicable).
9. Citations are returned.

---

# Hallucination Prevention

The prompt instructs Gemini to answer **only** from the retrieved document context.

If sufficient information is not available, the model returns:

> "The uploaded documents do not contain enough information to answer this question."

This prevents unsupported answers.

---

# Citations

Every answer includes citations containing:

- Source document
- Page number
- Chunk ID
- Supporting text snippet

This allows users to verify the generated answer against the original document.

---

# Multilingual Support

The system supports multilingual questions through:

- Language detection (`langdetect`)
- Translation to English (`deep-translator`)
- Retrieval in English
- Translation of the final answer back into the user's original language

This allows users to ask questions in languages such as:

- English
- Hindi
- Marathi
- French
- Spanish
- and other supported languages

---

# Document Comparison

The `/contradict` endpoint compares two documents on a specified topic.

Workflow:

1. Retrieve relevant chunks from Document A.
2. Retrieve relevant chunks from Document B.
3. Provide both contexts to Gemini.
4. Generate a comparison indicating whether the documents:
   - Agree
   - Conflict
   - Partially Agree

---

# API Endpoints

## POST `/ask`

Request

```json
{
    "question": "What is Generative AI?",
    "rerank": false
}
```

Response

```json
{
    "question": "...",
    "language": "en",
    "answer": "...",
    "citations": [
        {
            "document": "...",
            "page": 3,
            "chunk_id": 42,
            "snippet": "..."
        }
    ],
    "confidence": 0.81,
    "needs_review": false
}
```

`rerank` is optional (default `false`); see [Cross-Encoder Reranker](#2-cross-encoder-reranker).

---

## GET `/documents`

Returns the list of indexed documents, used by the Streamlit UI to populate
the document dropdowns in the compare tab.

```json
{
    "documents": ["Chemistry.pdf", "ICSICE.pdf", "..."]
}
```

---

## POST `/contradict`

Request

```json
{
    "document1": "ICSICE.pdf",
    "document2": "Foundations of GenAI & Tools.pdf",
    "topic": "Generative AI"
}
```

Response

```json
{
    "comparison": "Partial Agreement ..."
}
```

---

# Streamlit UI

The project includes a Streamlit interface that allows users to:

- Ask questions without Postman
- View generated answers
- View supporting citations
- Compare two documents

---

# Technologies Used

- Python
- FastAPI
- Streamlit
- FAISS
- Sentence Transformers
- LangChain Text Splitter
- Google Gemini 2.5 Flash
- PyMuPDF
- LangDetect
- Deep Translator

---

# Installation

```bash
git clone <repository-url>

cd Rag-Assignment

python -m venv .venv

source .venv/bin/activate
```

Windows

```bash
.venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Build the Vector Store

```bash
python app/ingest.py
```

---

# Run FastAPI

```bash
uvicorn app.main:app --reload --port 8001
```

---

# Run Streamlit

```bash
streamlit run ui/app.py
```

---


# Author

**Sanika Gajarishi**

