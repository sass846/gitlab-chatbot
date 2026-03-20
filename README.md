# GitLab RAG Chatbot Backend

## Overview

This backend was built for the assignment of creating a Retrieval-Augmented Generation chatbot for GitLab's handbook and direction pages. Its responsibility is to clean and index the documentation, retrieve the most relevant passages for a user query, and generate grounded answers through a FastAPI service.

The backend is designed around one core principle: answer quality depends first on retrieval quality. Because of that, most of the backend work focused on document preparation, retrieval design, reranking, and conversational query handling before generation.

## What This Backend Does

- Cleans and chunks GitLab markdown documentation
- Builds a vector index for semantic retrieval
- Builds a BM25 index for keyword retrieval
- Uses hybrid retrieval to improve recall
- Reranks retrieved passages to improve precision
- Rewrites follow-up questions using conversation history
- Routes queries through different retrieval strategies
- Generates grounded answers through a FastAPI `/api/chat` endpoint

## Architecture

The backend consists of four major stages:

1. Document preprocessing
2. Indexing
3. Retrieval and reranking
4. Answer generation

### 1. Document Preprocessing

The source documents are markdown files from GitLab handbook and direction pages. Before indexing, they are cleaned and structured in [`index.ipynb`](./index.ipynb).

This preprocessing includes:

- parsing frontmatter
- cleaning markdown noise
- preserving title, description, path, and section hierarchy
- chunking by heading structure
- splitting oversized sections into smaller sub-chunks
- filtering out weak or low-value text

This step was necessary because raw markdown is noisy. Better document quality leads directly to better retrieval quality.

### 2. Indexing

I experimented with multiple embedding models before deciding on the final setup.

- MiniLM was used first as a lightweight baseline
- `bge-large` performed best in retrieval quality
- `bge-base` was selected for the final system

`bge-large` gave the strongest results, but it requires stronger infrastructure and is less practical for constrained or private deployment. `bge-base` performed well enough while remaining CPU-friendly, so it provided the best balance between quality and deployability.

The final vector index uses:

- `BAAI/bge-base-en-v1.5` for embeddings
- Chroma as the persistent vector store

### 3. Retrieval and Reranking

Dense retrieval alone was not enough for handbook-style content, because many GitLab questions depend on exact terminology and keyword overlap.

To improve retrieval quality, I added:

- BM25 retrieval for sparse lexical matching
- hybrid retrieval combining BM25 and vector search
- deduplication of overlapping results
- reranking using `BAAI/bge-reranker-base`

This design improved both recall and precision:

- hybrid retrieval increased the chance of finding relevant evidence
- reranking improved the quality of the final context passed to the LLM

### 4. Query Routing and History Handling

Not all questions should follow the same retrieval strategy. To address that, I added a router with three query types:

- `DIRECT` for simple direct questions
- `MULTI` for broader questions needing multiple alternate phrasings
- `DECOMPOSE` for multi-part questions that should be split into subqueries

I also added history-aware rewriting for follow-up questions. The latest user message is rewritten into a standalone query using recent conversation context before retrieval.

These two decisions were important because:

- query routing improves retrieval strategy selection
- history-aware rewriting improves conversational follow-up handling

### 5. Answer Generation

After retrieval and reranking, the top supporting passages are formatted into context and passed to Gemini through `ChatGoogleGenerativeAI`.

The answer prompt is designed to:

- start with a concise definition
- continue with bullet points
- ignore irrelevant context
- rely on retrieved documents for factual grounding
- explicitly admit uncertainty if the evidence is insufficient

This keeps answers readable and grounded in GitLab documentation instead of free-form model knowledge.

## Tech Stack

- Python
- FastAPI
- LangChain
- Chroma
- Hugging Face embeddings
- BM25
- Sentence Transformers CrossEncoder
- Gemini via `langchain_google_genai`

## API

### `POST /api/chat`

Accepts:

```json
{
  "messages": [
    { "role": "user", "content": "What are GitLab's core values?" }
  ]
}
```

Returns:

```json
{
  "role": "assistant",
  "content": "Grounded answer text",
  "steps": {
    "standalone_question": "Rewritten question",
    "query_type": "DIRECT",
    "subqueries": [],
    "num_docs": 5
  },
  "docs": [
    {
      "section": "Values",
      "content": "Retrieved passage",
      "source": "path/to/file.md"
    }
  ]
}
```

## Project Files

- [`main.py`](./main.py): FastAPI app and RAG pipeline
- [`index.ipynb`](./index.ipynb): preprocessing, chunking, indexing experiments
- `chroma_gitlab/`: Chroma vector database
- `docs_bm25.pkl`: BM25 document corpus
- `gitlab_chunks.pkl`: processed chunk data

## Why the Backend Looks This Way

The final backend design is the result of a sequence of practical decisions:

- document cleanup was added because raw markdown reduced retrieval quality
- heading-aware chunking was used because GitLab documentation is hierarchical
- `bge-base` was chosen because it balanced quality and deployability
- BM25 was added because dense retrieval alone was not enough
- reranking was added because hybrid retrieval still needed precision improvement
- query routing was added because not all questions behave the same way
- history-aware rewriting was added because follow-up questions are often underspecified
- FastAPI was chosen because the backend needed to be a lightweight, stateless API service

## Setup

### 1. Install dependencies

```bash
cd gitlab-chatbot
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file with the required model credentials, for example:

```env
GOOGLE_API_KEY=your_key_here
```

### 3. Ensure local data files exist

The backend expects these assets to be present:

- `chroma_gitlab/`
- `docs_bm25.pkl`

### 4. Run the backend

```bash
python main.py
```

The API will start on:

```text
http://localhost:8000
```

## Hugging Face Spaces Deployment

This backend is prepared for a Hugging Face Docker Space.

### What to upload

Push this backend directory as the contents of your Docker Space repository, including:

- `main.py`
- `requirements.txt`
- `Dockerfile`
- `README.md`
- `docs_bm25.pkl`
- `chroma_gitlab/`

You do not need to upload:

- `.venv/`
- `__pycache__/`
- `index.ipynb`
- `chroma_db2.zip`
- `gitlab_chunks.pkl`

### Space settings

In your Hugging Face Space:

- choose `Docker` as the SDK
- keep `CPU Basic` unless startup memory becomes an issue
- add the secret `GOOGLE_API_KEY`

### Optional environment variables

These are already supported by the app if you need them later:

```env
PORT=7860
CHROMA_PERSIST_DIR=/app/chroma_gitlab
BM25_DOCS_PATH=/app/docs_bm25.pkl
ALLOWED_ORIGINS=*
```

### Runtime endpoints

After deployment, the important endpoints are:

- `/health`
- `/api/chat`

## Notes

- The backend is stateless. It does not store user accounts or conversation history permanently.
- It expects the client to send recent conversation messages with each request.
- This keeps the RAG service focused on inference and avoids mixing application state with retrieval logic.

## Conclusion

The backend was designed to maximize grounded answer quality under practical deployment constraints. Instead of relying only on one embedding model and a basic vector search, it uses cleaned markdown, structured chunking, hybrid retrieval, reranking, query routing, and history-aware rewriting to produce stronger answers over GitLab handbook and direction content.
