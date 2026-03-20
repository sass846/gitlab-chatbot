import os
import pickle
from pathlib import Path
from typing import List, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from sentence_transformers import CrossEncoder

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = Path(os.environ.get("CHROMA_PERSIST_DIR", BASE_DIR / "chroma_gitlab"))
BM25_DOCS_PATH = Path(os.environ.get("BM25_DOCS_PATH", BASE_DIR / "docs_bm25.pkl"))
PORT = int(os.environ.get("PORT", "8000"))
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]


#model and store init
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    temperature=0
)

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": "cpu"}
)

vectorstore = Chroma(
    collection_name="gitlab-final",
    embedding_function=embeddings,
    persist_directory=str(CHROMA_DIR)
)

vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

with open(BM25_DOCS_PATH, "rb") as f:
    all_docs_bm25 = pickle.load(f)

bm25_retriever = BM25Retriever.from_documents(all_docs_bm25)
bm25_retriever.k = 8

reranker_model = CrossEncoder("BAAI/bge-reranker-base", device="cpu")


#routing setup
class QueryRouter(BaseModel):
    """Routes a user query to the appropriate search strategy."""
    route: Literal["DIRECT", "MULTI", "DECOMPOSE"] = Field(
        description=(
            "DIRECT: Simple, direct lookup for a single concept. "
            "MULTI: A single complex concept that needs broader alternative search phrasing. "
            "DECOMPOSE: The user is asking multiple distinct, independent questions."
        )
    )
    queries: List[str] = Field(
        description=(
            "If route is DIRECT, leave empty. "
            "If route is MULTI, provide 3 alternative search phrasings. "
            "If route is DECOMPOSE, break the prompt into 2-3 independent, fully contextualized subqueries."
        ),
        default_factory=list
    )

structured_router_llm = llm.with_structured_output(QueryRouter)

ROUTER_PROMPT = ChatPromptTemplate.from_template("""
You are an expert search traffic router for GitLab documentation. 
Analyze the user's query and decide how to process it.

Query: {query}
""")

FOLLOWUP_PROMPT = ChatPromptTemplate.from_template("""
Given the conversation and a follow-up question, rewrite it into a standalone question.

Rules:
- Preserve original meaning
- Be specific
- Do NOT answer the question

Conversation:
{history}

Follow-up:
{question}

Standalone question:
""")

#helper functions
def get_text(response):
    content = response.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        ).strip()
    return str(content).strip()

def rewrite_query(question, history):
    if not history:
        return question

    # Take last few turns only to avoid token explosion
    history_text = "\n".join(
        f"{h['role']}: {h['content']}"
        for h in history[-4:]
    )

    chain = FOLLOWUP_PROMPT | llm
    response = chain.invoke({"history": history_text, "question": question})
    return get_text(response)

def route_query(query: str) -> QueryRouter:
    """Uses structured output to deterministically route and expand/decompose the query."""
    chain = ROUTER_PROMPT | structured_router_llm
    return chain.invoke({"query": query})

def rerank(query, docs, top_k=3):
    if not docs:
        return []
    pairs = [(query, d.page_content) for d in docs]
    scores = reranker_model.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]

def dedup_docs(docs):
    seen = set()
    unique = []
    for d in docs:
        key = (
            d.metadata.get("source", "") +
            d.metadata.get("section", "") +
            d.page_content[:50]
        )
        if key not in seen:
            seen.add(key)
            unique.append(d)
    return unique


#retrieval logic
def hybrid_retrieve(query):
    bm25_docs = bm25_retriever.invoke(query)
    vector_docs = vector_retriever.invoke(query)
    
    # Simple deduplication for the immediate hybrid results
    seen = set()
    combined = []
    for doc in bm25_docs + vector_docs:
        key = doc.metadata.get("source", "") + doc.metadata.get("section", "") + doc.page_content[:50]
        if key not in seen:
            seen.add(key)
            combined.append(doc)
            
    return combined

def retrieve_docs(query: str, top_k=3):
    """Orchestrates retrieval based on the structured router's decision."""
    routing_decision = route_query(query)
    
    all_retrieved_docs = []
    
    if routing_decision.route == "DIRECT" or not routing_decision.queries:
        all_retrieved_docs = hybrid_retrieve(query)
    
    else:
        # For MULTI or DECOMPOSE, we iterate through the generated subqueries
        for subquery in routing_decision.queries:
            docs = hybrid_retrieve(subquery)
            # Rerank individually to ensure the best docs for *this specific subquery* bubble up
            best_sub_docs = rerank(subquery, docs, top_k=2) 
            all_retrieved_docs.extend(best_sub_docs)

    # Final deduplication and reranking against the original main query
    unique_docs = dedup_docs(all_retrieved_docs)
    final_docs = rerank(query, unique_docs, top_k=top_k)
    
    return final_docs, routing_decision

def format_docs(docs):
    cleaned = []
    for d in docs:
        text = d.page_content.strip()
        if not text or "Document(" in text:
            continue
        cleaned.append(text[:500])
    return "\n\n".join(cleaned)

#rag pipelines
qa_prompt = ChatPromptTemplate.from_template("""
You are an expert assistant answering questions from GitLab documentation.

Instructions:
- Start with a concise definition
- Then give bullet points
- Ignore irrelevant context
- Use the conversation history for context if needed, but rely on the provided documents for facts.
- If unsure, say: "I don't know based on the provided documents."

Conversation History:
{history}

Context:
{context}

Question:
{question}

Answer:
""")

def rag_pipeline(question: str):
    docs, _ = retrieve_docs(question, top_k=3)
    context = format_docs(docs)

    chain = (
        {
            "context": lambda _: context,
            "question": RunnablePassthrough()
        }
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(question)
    return answer, docs

def rag_chat_with_steps(question: str, history: list):
    standalone_q = rewrite_query(question, history)
    docs, routing_decision = retrieve_docs(standalone_q, top_k=5)
    context = format_docs(docs)

    history_text = "\n".join(f"{h['role']}: {h['content']}" for h in history[-4:]) if history else "No previous history."

    chain = (
        {
            "history": lambda _: history_text,
            "context": lambda _: context,
            "question": RunnablePassthrough()
        }
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(standalone_q)

    return {
        "answer": answer,
        "steps": {
            "standalone_question": standalone_q,
            "query_type": routing_decision.route,
            "subqueries": routing_decision.queries,
            "num_docs": len(docs),
            "docs_preview": [
                {
                    "section": d.metadata.get("section"),
                    "content": d.page_content[:200]
                }
                for d in docs
            ]
        },
        "docs": docs
    }

#execution
class ChatRequest(BaseModel):
    messages: List[dict]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def healthcheck():
    return {
        "status": "ok",
        "chroma_dir": str(CHROMA_DIR),
        "bm25_docs_path": str(BM25_DOCS_PATH),
    }

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    messages = request.messages
    if not messages:
        return {"error": "No messages provided"}
    
    query = messages[-1].get("content", "")
    history = messages[:-1]
    
    try:
        result = rag_chat_with_steps(query, history)
        
        docs = [
            {
                "section": d.metadata.get("section", ""),
                "content": d.page_content,
                "source": d.metadata.get("source", "")
            }
            for d in result.get("docs", [])
        ]
        
        return {
            "role": "assistant",
            "content": result.get("answer", ""),
            "steps": result.get("steps", {}),
            "docs": docs
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
