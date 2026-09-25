import os
import re
import json
from typing import TypedDict, List, Optional
from pydantic import BaseModel, Field
import chromadb
from chromadb.utils import embedding_functions
from langgraph.graph import StateGraph, START, END

# Toggle environment flag: defaults to "1" (Offline Deterministic Mock Mode)
MOCK_LLM = os.getenv("MOCK_LLM", "1") == "1"


# -----------------------------------------------------------------------------
# 1. STRUCTURED OUTPUT SCHEMA (PYDANTIC)
# -----------------------------------------------------------------------------
class SupportAssistantResponse(BaseModel):
    answer: str = Field(..., description="The assistant's grounded reply.")
    sources: List[str] = Field(default_factory=list, description="IDs of source chunks used.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0.")


# -----------------------------------------------------------------------------
# 2. STRUCTURED PROMPT TEMPLATE (Role-Context-Task-Format-Length)
# -----------------------------------------------------------------------------
STRUCTURED_PROMPT_TEMPLATE = """### ROLE
You are Zepto's official AI Support Assistant, providing accurate, grounded assistance on store policies.

### CONTEXT
{context}

### TASK
Answer the user's inquiry accurately using ONLY the information provided in the CONTEXT above.
NEGATIVE CONSTRAINT: Do not answer using information not present in the provided context. If the context does not contain sufficient details to answer, state: "I cannot find this information in Zepto's official policy documentation."

### FEW-SHOT EXAMPLE
Context:
[doc_08]: Zepto customer support is available via in-app chat 24/7. Phone support is not offered.
User Query: Can I call Zepto support on the phone?
Output Format (JSON):
{{
  "answer": "No, Zepto does not offer phone support. Support is available 24/7 exclusively via in-app chat and email.",
  "sources": ["doc_08"],
  "confidence": 1.0
}}

### USER QUERY
{query}

### FORMAT & LENGTH
Provide your response strictly in valid JSON adhering to this schema:
{{
  "answer": "<2-4 sentences max>",
  "sources": ["<doc_id>", ...],
  "confidence": <float between 0.0 and 1.0>
}}
"""


# -----------------------------------------------------------------------------
# 3. LANGGRAPH STATE SCHEMA
# -----------------------------------------------------------------------------
class AssistantState(TypedDict):
    query: str
    intent: Optional[str]
    retrieved_docs: List[dict]
    response: Optional[SupportAssistantResponse]


# ChromaDB Accessor
DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path=DB_DIR)
collection = chroma_client.get_or_create_collection(name="zepto_policies", embedding_function=emb_fn)

# -----------------------------------------------------------------------------
# 4. GRAPH NODES
# -----------------------------------------------------------------------------
POLICY_KEYWORDS = [
    "delivery", "return", "refund", "membership",
    "tracking", "cancel", "gift card", "support hours"
]


def classify_intent(state: AssistantState) -> dict:
    """Classifies the query into policy_question or general_question."""
    query = state["query"].lower()

    if MOCK_LLM:
        # Mock mode: Keyword-heuristic classification (No network or LLM calls)
        is_policy = any(kw in query for kw in POLICY_KEYWORDS)
        intent = "policy_question" if is_policy else "general_question"
    else:
        # Optional real LLM branch (e.g. Groq free tier)
        intent = "policy_question" if any(kw in query for kw in POLICY_KEYWORDS) else "general_question"

    return {"intent": intent}


def retrieve_and_answer(state: AssistantState) -> dict:
    """Retrieves top-3 chunks from ChromaDB and forms grounded response."""
    query = state["query"]

    # Cosine similarity retrieval (Runs locally for both mock and real modes)
    results = collection.query(query_texts=[query], n_results=3)

    retrieved_chunks = []
    if results and "documents" in results and results["documents"]:
        for doc_text, doc_id in zip(results["documents"][0], results["ids"][0]):
            retrieved_chunks.append({"id": doc_id, "text": doc_text})

    if MOCK_LLM:
        # Deterministic mock generation: Canned template using top chunk excerpt
        if retrieved_chunks:
            top_chunk_snippet = retrieved_chunks[0]["text"][:200]
            answer_text = f"Based on the retrieved context: {top_chunk_snippet}"
            sources = [c["id"] for c in retrieved_chunks]
        else:
            answer_text = "Based on the retrieved context: No relevant policy found."
            sources = []

        validated_resp = SupportAssistantResponse(
            answer=answer_text,
            sources=sources,
            confidence=1.0
        )
    else:
        # Optional real-LLM branch with retry loop on validation failure
        context_str = "\n\n".join([f"[{c['id']}]: {c['text']}" for c in retrieved_chunks])
        prompt = STRUCTURED_PROMPT_TEMPLATE.format(context=context_str, query=query)
        validated_resp = call_llm_with_retry(prompt, expected_sources=[c["id"] for c in retrieved_chunks])

    return {"retrieved_docs": retrieved_chunks, "response": validated_resp}


def direct_answer(state: AssistantState) -> dict:
    """Answers out-of-scope general inquiries with canned message."""
    if MOCK_LLM:
        validated_resp = SupportAssistantResponse(
            answer="I can only answer questions about Zepto policies right now.",
            sources=[],
            confidence=1.0
        )
    else:
        validated_resp = SupportAssistantResponse(
            answer="I can only answer questions about Zepto policies right now.",
            sources=[],
            confidence=0.9
        )
    return {"response": validated_resp}


# Optional real-LLM caller with retry mechanism
def call_llm_with_retry(prompt: str, expected_sources: List[str], retries: int = 2) -> SupportAssistantResponse:
    try:
        from groq import Groq
        client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
        for attempt in range(retries + 1):
            res = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            raw_content = res.choices[0].message.content
            # Extract JSON substring
            match = re.search(r"\{.*\}", raw_content, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                return SupportAssistantResponse(**data)
            prompt += "\nERROR: Output was not valid JSON. Ensure strict adherence to the JSON schema."
    except Exception:
        pass
    return SupportAssistantResponse(
        answer="Error processing policy query through real LLM backend.",
        sources=expected_sources,
        confidence=0.0
    )


# -----------------------------------------------------------------------------
# 5. STATEGRAPH ROUTING & COMPILATION
# -----------------------------------------------------------------------------
def route_intent(state: AssistantState) -> str:
    return "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"


builder = StateGraph(AssistantState)
builder.add_node("classify_intent", classify_intent)
builder.add_node("retrieve_and_answer", retrieve_and_answer)
builder.add_node("direct_answer", direct_answer)

builder.add_edge(START, "classify_intent")
builder.add_conditional_edges(
    "classify_intent",
    route_intent,
    {
        "retrieve_and_answer": "retrieve_and_answer",
        "direct_answer": "direct_answer"
    }
)
builder.add_edge("retrieve_and_answer", END)
builder.add_edge("direct_answer", END)

support_agent_graph = builder.compile()