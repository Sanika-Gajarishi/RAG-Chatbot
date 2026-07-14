import json
import re

from .rag import retrieve, build_context

SYSTEM_PROMPT = """
You are an expert document comparison assistant.

Compare ONLY the information contained in the two supplied document contexts.

Your task:

1. Determine whether the documents:
   - Agree
   - Conflict
   - Partially Agree

2. Explain your reasoning.

3. Mention the important differences.

4. Do NOT use outside knowledge.

Return your answer in this JSON format:

{
    "status":"Agree | Conflict | Partial Agreement",
    "reason":"Short explanation",
    "summary":"Detailed comparison"
}
"""


def compare_documents(model, document1, document2, topic):

    # Retrieve relevant chunks from each document
    doc1_results = retrieve(
        query=topic,
        top_k=5,
        document=document1
    )

    doc2_results = retrieve(
        query=topic,
        top_k=5,
        document=document2
    )

    # Handle missing information
    if len(doc1_results) == 0:
        return {
            "status": "Not Enough Information",
            "reason": f"No relevant information found in {document1}.",
            "summary": ""
        }

    if len(doc2_results) == 0:
        return {
            "status": "Not Enough Information",
            "reason": f"No relevant information found in {document2}.",
            "summary": ""
        }

    context1 = build_context(doc1_results)
    context2 = build_context(doc2_results)

    prompt = f"""
{SYSTEM_PROMPT}

========================
DOCUMENT 1
========================

Filename:
{document1}

Context:

{context1}

========================
DOCUMENT 2
========================

Filename:
{document2}

Context:

{context2}

========================
TOPIC
========================

{topic}
"""

    response = model.generate_content(prompt)

    return _parse_comparison(response.text)


def _parse_comparison(raw_text: str):
    """
    Gemini is asked to return JSON, but LLMs occasionally wrap it in
    markdown fences or add stray text. Try to parse it cleanly; if that
    fails, fall back to returning the raw text under "summary" so the
    caller never crashes on a formatting slip.
    """

    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {
            "status": "Unknown",
            "reason": "Model response was not valid JSON.",
            "summary": raw_text
        }