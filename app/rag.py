import os
import pickle
import math
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ==========================================================
# Configuration
# ==========================================================

INDEX_PATH = "data/faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Below this confidence (0-1), the answer is flagged for human review
# instead of being served automatically.
CONFIDENCE_THRESHOLD = 0.35

# ==========================================================
# Load Embedding Model
# ==========================================================

print("Loading embedding model...")

model = SentenceTransformer(EMBEDDING_MODEL)

# ==========================================================
# Reranker (lazy-loaded, stretch goal)
# ==========================================================
# The cross-encoder scores (query, chunk) pairs directly, which is slower
# than the bi-encoder FAISS search but much more precise. We therefore use
# FAISS to pull a wider net of candidates, then let the cross-encoder
# re-sort just those candidates. The model is only loaded the first time
# it's actually needed so `retrieve(..., rerank=False)` stays fast and
# doesn't require the reranker download at all.

_reranker = None


def _get_reranker():
    global _reranker

    if _reranker is None:
        from sentence_transformers import CrossEncoder
        print("Loading reranker model (first use only)...")
        _reranker = CrossEncoder(RERANKER_MODEL)

    return _reranker

# ==========================================================
# Load FAISS Index
# ==========================================================

index_path = os.path.join(INDEX_PATH, "faiss.index")

if not os.path.exists(index_path):
    raise FileNotFoundError(
        "FAISS index not found. Please run ingest.py first."
    )

index = faiss.read_index(index_path)

# ==========================================================
# Load Metadata
# ==========================================================

metadata_path = os.path.join(INDEX_PATH, "metadata.pkl")

if not os.path.exists(metadata_path):
    raise FileNotFoundError(
        "metadata.pkl not found. Please run ingest.py first."
    )

with open(metadata_path, "rb") as f:
    metadata = pickle.load(f)

print(f"Loaded {len(metadata)} chunks.")

# ==========================================================
# Retrieve Chunks
# ==========================================================


def retrieve(
    query: str,
    top_k: int = 5,
    document: str | None = None,
    rerank: bool = False,
    candidate_k: int = 20
):
    """
    Retrieve the most relevant chunks.

    Args:
        query: User question
        top_k: Number of chunks to return
        document: Optional document filter
        rerank: If True, pull `candidate_k` chunks from FAISS and re-sort
            them with a cross-encoder before truncating to top_k. This is
            slower but noticeably more precise, especially when the
            bi-encoder retrieves several plausible-looking chunks.
        candidate_k: How many bi-encoder candidates to hand to the
            reranker when rerank=True.

    Returns:
        List of retrieved chunks, each with a "score" (bi-encoder L2
        distance, lower = closer) and, when reranked, a "rerank_score"
        (cross-encoder relevance logit, higher = more relevant).
    """

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    # Search entire index
    distances, indices = index.search(
        query_embedding,
        len(metadata)
    )

    fetch_limit = candidate_k if rerank else top_k

    results = []

    for distance, idx in zip(distances[0], indices[0]):

        if idx == -1:
            continue

        chunk = metadata[idx]

        # Filter by document if requested
        if document is not None:

            if chunk["document"] != document:
                continue

        results.append({

            "document": chunk["document"],

            "page": chunk["page"],

            "chunk_id": chunk["chunk_id"],

            "score": round(float(distance), 4),

            "snippet": chunk["text"][:250],

            "text": chunk["text"]

        })

        if len(results) >= fetch_limit:
            break

    if rerank and results:
        results = rerank_results(query, results)

    return results[:top_k]


# ==========================================================
# Reranking
# ==========================================================


def rerank_results(query: str, results: list):
    """
    Re-score a list of candidate chunks with a cross-encoder and
    re-sort them by relevance (most relevant first).
    """

    reranker = _get_reranker()

    pairs = [[query, result["text"]] for result in results]

    scores = reranker.predict(pairs)

    for result, score in zip(results, scores):
        result["rerank_score"] = round(float(score), 4)

    results.sort(key=lambda r: r["rerank_score"], reverse=True)

    return results


# ==========================================================
# Confidence Scoring
# ==========================================================


def compute_confidence(results):
    """
    Turn retrieval scores into a single 0-1 confidence value for the
    top-ranked chunk, used to decide whether an answer should be gated
    for human review.

    - If the results were reranked, the cross-encoder logit is squashed
      through a sigmoid (a natural fit for a relevance logit).
    - Otherwise, we fall back to the bi-encoder L2 distance: smaller
      distance -> higher confidence. All-MiniLM-L6-v2 distances typically
      fall in the 0-2 range, so exp(-distance) maps that onto (0, 1].
    """

    if not results:
        return 0.0

    top = results[0]

    if "rerank_score" in top:
        return round(1 / (1 + math.exp(-top["rerank_score"])), 4)

    distance = top["score"]
    confidence = math.exp(-distance)
    return round(min(max(confidence, 0.0), 1.0), 4)


# ==========================================================
# Build Context for Gemini
# ==========================================================


def build_context(results):

    context = []

    for chunk in results:

        context.append(

            f"""
DOCUMENT: {chunk['document']}
PAGE: {chunk['page']}
CHUNK: {chunk['chunk_id']}

CONTENT:
{chunk['text']}
"""

        )

    return "\n\n".join(context)


# ==========================================================
# List Available Documents
# ==========================================================


def list_documents():

    docs = sorted(
        list(
            {
                item["document"]
                for item in metadata
            }
        )
    )

    return docs


# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    print("\nIndexed Documents\n")

    for doc in list_documents():
        print("-", doc)

    print()

    while True:

        query = input("Ask a question (or type exit): ")

        if query.lower() == "exit":
            break

        results = retrieve(query)

        print("\n")

        for i, result in enumerate(results, start=1):

            print("=" * 80)

            print(f"Rank      : {i}")

            print(f"Document  : {result['document']}")

            print(f"Page      : {result['page']}")

            print(f"Chunk ID  : {result['chunk_id']}")

            print(f"Distance  : {result['score']}")

            print("\nSnippet:\n")

            print(result["snippet"])

            print("\n")