import os
import google.generativeai as genai

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .rag import (
    retrieve,
    build_context,
    list_documents,
    compute_confidence,
    CONFIDENCE_THRESHOLD
)
from .translator import (
    translate_to_english,
    translate_from_english
)
from .contradiction import compare_documents

# ==================================================
# Load Environment Variables
# ==================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in .env file.")

genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("gemini-2.5-flash")

# ==================================================
# FastAPI App
# ==================================================

app = FastAPI(
    title="Document Q&A with Citations",
    description="RAG-based Document Question Answering System",
    version="1.0"
)

# ==================================================
# Root Endpoint
# ==================================================

@app.get("/")
def home():

    return {
        "message": "Document Q&A API is running successfully.",
        "docs": "/docs"
    }

# ==================================================
# Request Models
# ==================================================

class Question(BaseModel):
    question: str
    rerank: bool = False


class ContradictionRequest(BaseModel):
    document1: str
    document2: str
    topic: str

# ==================================================
# Prompt
# ==================================================

PROMPT = """
You are a Document Question Answering Assistant.

Answer ONLY using the supplied document context.

Rules:

1. Never use outside knowledge.

2. If the answer is not present in the supplied context, reply exactly:

"The uploaded documents do not contain enough information to answer this question."

3. Do not invent facts.

4. Write the answer in clear paragraphs.

5. Explain the concept using only the provided context.

6. Do not mention document names inside the answer.

7. Do not use bullet points unless necessary.

8. Keep the answer professional and concise.
"""

# ==================================================
# Ask Endpoint
# ==================================================

@app.post("/ask")
def ask(question: Question):

    # ----------------------------------------
    # Translate Question
    # ----------------------------------------

    english_question, original_language = translate_to_english(
        question.question
    )

    # ----------------------------------------
    # Retrieve Relevant Chunks
    # ----------------------------------------

    retrieved = retrieve(
        english_question,
        rerank=question.rerank
    )

    # ----------------------------------------
    # Nothing Retrieved
    # ----------------------------------------

    if len(retrieved) == 0:

        return {

            "question": question.question,

            "language": original_language,

            "answer": "The uploaded documents do not contain enough information to answer this question.",

            "citations": [],

            "confidence": 0.0,

            "needs_review": True

        }

    # ----------------------------------------
    # Confidence Score (stretch: human-in-the-loop gate)
    # ----------------------------------------

    confidence = compute_confidence(retrieved)
    needs_review = confidence < CONFIDENCE_THRESHOLD

    # ----------------------------------------
    # Build Context
    # ----------------------------------------

    context = build_context(
        retrieved
    )

    final_prompt = f"""
{PROMPT}

=========================
DOCUMENT CONTEXT
=========================

{context}

=========================
QUESTION
=========================

{english_question}

Provide a detailed answer in paragraphs.

Start with a short introduction.

Then explain the concept.

End with a short conclusion.

Remember:
Answer ONLY from the supplied document context.
"""
        # ----------------------------------------
    # Generate Answer using Gemini
    # ----------------------------------------

    try:

        response = model.generate_content(
            final_prompt
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Gemini API Error: {str(e)}"
        )

    # ----------------------------------------
    # Translate Answer Back
    # ----------------------------------------

    final_answer = translate_from_english(
        response.text,
        original_language
    )

    # ----------------------------------------
    # Build Citations
    # ----------------------------------------

    citations = []

    for chunk in retrieved:

        citation = {

            "document": chunk["document"],

            "page": chunk["page"],

            "chunk_id": chunk["chunk_id"],

            "snippet": chunk["snippet"]

        }

        if "rerank_score" in chunk:
            citation["rerank_score"] = chunk["rerank_score"]

        citations.append(citation)

    # ----------------------------------------
    # Return Response
    # ----------------------------------------

    return {

        "question": question.question,

        "language": original_language,

        "answer": final_answer,

        "citations": citations,

        "confidence": confidence,

        "needs_review": needs_review

    }


# ==================================================
# Contradict Endpoint
# ==================================================

@app.post("/contradict")
def contradict(request: ContradictionRequest):

    try:

        result = compare_documents(

            model=model,

            document1=request.document1,

            document2=request.document2,

            topic=request.topic

        )

        return {

            "document1": request.document1,

            "document2": request.document2,

            "topic": request.topic,

            "comparison": result

        }

    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=f"Comparison Error: {str(e)}"

        )
    
@app.get("/documents")
def get_documents():
    return {
        "documents": list_documents()
    }