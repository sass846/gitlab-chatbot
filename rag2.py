from dotenv import load_dotenv
load_dotenv()

import pickle
from collections import defaultdict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from sentence_transformers import CrossEncoder

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    temperature=0
)

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5"
)

vectorstore = Chroma(
    collection_name="gitlab-final",
    embedding_function=embeddings,
    persist_directory="./chroma_gitlab"
)

vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

with open("docs_bm25.pkl", "rb") as f:
    all_docs_bm25 = pickle.load(f)

bm25_retriever = BM25Retriever.from_documents(all_docs_bm25)
bm25_retriever.k = 8

reranker_model = CrossEncoder("BAAI/bge-reranker-base")

def rerank(query, docs, top_k=3):
    if not docs:
        return []

    pairs = [(query, d.page_content) for d in docs]
    scores = reranker_model.predict(pairs)

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]

# reranker = RunnableLambda(rerank_fn)
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


#prompts

MULTI_QUERY_PROMPT = """
You are a search expert.

Generate 3 alternative search queries for the question.

Guidelines:
- Use terminology likely found in documentation
- Include related concepts or frameworks
- Keep each query short (one line)
- No explanations

Question:
{query}

Queries:
"""

ROUTER_PROMPT = """
Classify the query into one of:

- SIMPLE → direct lookup
- MULTI → needs broader retrieval
- COMPLEX → multiple questions

Query:
{query}

Answer ONLY one word.
"""

DECOMPOSE_PROMPT = """
Break this query into smaller independent questions.

Query:
{query}

Return each question on a new line.
"""

FOLLOWUP_PROMPT = """
Given the conversation and a follow-up question,
rewrite it into a standalone question.

Rules:
- Preserve original meaning
- Be specific
- Do NOT answer the question

Conversation:
{history}

Follow-up:
{question}

Standalone question:
"""

def rewrite_query(question, history):
    if not history:
        return question

    # take last few turns only (avoid token explosion)
    history_text = "\n".join(
        f"{h['role']}: {h['content']}"
        for h in history[-4:]
    )

    response = llm.invoke(
        FOLLOWUP_PROMPT.format(
            history=history_text,
            question=question
        )
    )

    return get_text(response)


def get_text(response):
    content = response.content

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        # handle Gemini structured output
        return "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        ).strip()

    return str(content).strip()

# llm functions
def expand_query(query):
    response = llm.invoke(MULTI_QUERY_PROMPT.format(query=query))
    text = get_text(response)

    queries = [q.strip("- ").strip() for q in text.split("\n") if q.strip()]
    queries = [q for q in queries if len(q.split()) < 12]

    return [query] + queries[:3]

def classify_query(query):
    response = llm.invoke(ROUTER_PROMPT.format(query=query))
    return get_text(response).strip().upper().split()[0]

def decompose_query(query):
    response = llm.invoke(DECOMPOSE_PROMPT.format(query=query))
    text = get_text(response)

    return [q.strip("- ").strip() for q in text.split("\n") if q.strip()]


def hybrid_retrieve(query):
    bm25_docs = bm25_retriever.invoke(query)
    vector_docs = vector_retriever.invoke(query)

    seen = set()
    combined = []

    for doc in bm25_docs + vector_docs:
        key = doc.metadata.get("source", "") + doc.metadata.get("section", "") + doc.page_content[:50]

        if key not in seen:
            seen.add(key)
            combined.append(doc)

    return combined


def multi_retrieve(query):
    queries = expand_query(query)

    doc_scores = defaultdict(float)
    doc_map = {}

    for q in queries:
        docs = hybrid_retrieve(q)

        for rank, d in enumerate(docs):
            key = d.metadata.get("source", "") + d.metadata.get("section", "") + d.page_content[:50]

            doc_scores[key] += 1 / (rank + 1)
            doc_map[key] = d

    ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

    return [doc_map[k] for k, _ in ranked]


def retrieve_docs(query):
    qtype = classify_query(query)

    if qtype == "SIMPLE":
        docs = hybrid_retrieve(query)

    elif qtype == "MULTI":
        docs = multi_retrieve(query)

    elif qtype == "COMPLEX":
        subqueries = decompose_query(query)
        docs = []

        for sq in subqueries:
            docs.extend(multi_retrieve(sq))

    else:
        docs = hybrid_retrieve(query)

    return rerank(query, docs, top_k=3)


def format_docs(docs):
    cleaned = []

    for d in docs:
        text = d.page_content.strip()

        if not text or "Document(" in text:
            continue

        cleaned.append(text[:500])

    return "\n\n".join(cleaned)

prompt = ChatPromptTemplate.from_template("""
You are an expert assistant answering questions from GitLab documentation.

Instructions:
- Start with a concise definition
- Then give bullet points
- Ignore irrelevant context
- If unsure, say: "I don't know based on the provided documents."

Context:
{context}

Question:
{question}

Answer:
""")



def rag_pipeline(question: str):
    docs = retrieve_docs(question)
    context = format_docs(docs)

    chain = (
        {
            "context": lambda _: context,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(question)

    return answer, docs

def rag_chat_with_steps(question: str, history: list):
    standalone_q = rewrite_query(question, history)

    qtype = classify_query(standalone_q)

    subqueries = []

    if qtype == "SIMPLE":
        docs = hybrid_retrieve(standalone_q)

    elif qtype == "MULTI":
        docs = multi_retrieve(standalone_q)

    elif qtype == "COMPLEX":
        subqueries = decompose_query(standalone_q)

        all_docs = []

        for sq in subqueries:
            sub_docs = multi_retrieve(sq)
            sub_docs = rerank(sq, sub_docs, top_k=2)
            all_docs.extend(sub_docs)

        docs = dedup_docs(all_docs)

        docs = docs[:5]

    else:
        docs = hybrid_retrieve(standalone_q)

    if qtype != "COMPLEX":
        docs = rerank(standalone_q, docs, top_k=5)

    context = format_docs(docs)

    chain = (
        {
            "context": lambda _: context,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(standalone_q)

    return {
        "answer": answer,
        "steps": {
            "standalone_question": standalone_q,
            "query_type": qtype,
            "subqueries": subqueries,
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


query = "What are GitLab values and what is procurement?"

answer, docs = rag_pipeline(query)

print("\nANSWER:\n")
print(answer)

print("\n\nTOP DOCS:\n")

for i, d in enumerate(docs):
    print(f"\n--- {i+1} ---")
    print("Section:", d.metadata.get("section"))
    print(d.page_content[:300])
