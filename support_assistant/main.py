from fastapi import FastAPI
from pydantic import BaseModel
from graph import support_agent_graph, SupportAssistantResponse

app = FastAPI(
    title="Zepto GenAI Support Assistant",
    description="Grounded policy Q&A service orchestrated with LangGraph & ChromaDB",
    version="1.0.0"
)

class QueryRequest(BaseModel):
    query: str

@app.post("/ask", response_model=SupportAssistantResponse)
def ask_policy(request: QueryRequest):
    initial_state = {
        "query": request.query,
        "intent": None,
        "retrieved_docs": [],
        "response": None
    }
    final_state = support_agent_graph.invoke(initial_state)
    return final_state["response"]

@app.get("/health")
def health_check():
    return {"status": "healthy"}