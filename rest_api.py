from typing import Any, Dict
import logging
import traceback

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Import existing agent components
from dynamic_rag_agent import (
    TrustAgent,
    QueryExtractorAgent,
    search_searxng,
    scrape_with_crawl4ai,
    Config,
    system_prompt,
)

from llama_index.core import VectorStoreIndex, Settings, Document
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from llama_index.core.storage import StorageContext
from llama_index.core.postprocessor import SimilarityPostprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Trust Agent REST API")


class Question(BaseModel):
    question: str


def _safe_get_response_text(resp: Any) -> str:
    # Try common response attributes
    try:
        if hasattr(resp, 'response'):
            return str(resp.response)
        return str(resp)
    except Exception:
        return ''


def answer_query(user_query: str) -> Dict[str, Any]:
    """Run the existing agent pipeline for a single query and return structured result.

    This mirrors the steps in `run_agent()` from `dynamic_rag_agent.py` but as a single-call function.
    """
    if not user_query or not user_query.strip():
        raise ValueError("Empty query provided")

    # Initialize components
    trust_agent = TrustAgent(Settings.llm)
    query_extractor = QueryExtractorAgent(Settings.llm)

    # Extract and enrich query into multiple focused queries
    enriched_queries = query_extractor.extract_and_enrich(user_query)

    # Search with multiple queries (each returns up to 10 results)
    search_results = search_searxng(enriched_queries, num_results_per_query=10)
    if not search_results:
        return {"answer": "", "error": "No sources found from search.", "sources": []}

    urls = [r.get("url", "") for r in search_results if r.get("url")]
    if not urls:
        return {"answer": "", "error": "No valid URLs returned by search.", "sources": []}

    # Trust filtering
    trusted_urls = trust_agent.filter_trusted(urls, user_query)
    if not trusted_urls:
        return {"answer": "", "error": "Insufficient reliable sources after trust filtering.", "sources": []}

    # Scrape trusted URLs
    scraped_docs = scrape_with_crawl4ai(trusted_urls, timeout_per_url=Config.SCRAPE_TIMEOUT_PER_URL)
    if not scraped_docs:
        return {"answer": "", "error": "Scraping failed for trusted sources.", "sources": trusted_urls}

    documents = []
    for doc in scraped_docs:
        text = doc.get("text", "")
        url = doc.get("url", "")
        if text:
            documents.append(Document(text=text, metadata={"url": url}))

    if not documents:
        return {"answer": "", "error": "No valid documents created from scraped content.", "sources": trusted_urls}

    # Vector store & index
    try:
        db = chromadb.PersistentClient(path=Config.CHROMA_DB_PATH)
        chroma_collection = db.get_or_create_collection(Config.CHROMA_COLLECTION_NAME)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

        postprocessor = SimilarityPostprocessor(similarity_cutoff=Config.SIMILARITY_CUTOFF)
        query_engine = index.as_query_engine(
            similarity_top_k=Config.SIMILARITY_TOP_K,
            node_postprocessors=[postprocessor],
            system_prompt=system_prompt,
            verbose=False,
        )

        resp = query_engine.query(user_query)
        answer_text = _safe_get_response_text(resp)
        sources = []
        try:
            if hasattr(resp, 'get_formatted_sources'):
                sources = resp.get_formatted_sources()
        except Exception:
            sources = []

        return {"answer": answer_text, "sources": sources, "trusted_urls": trusted_urls}
    except Exception as e:
        logger.error(f"Error during query processing: {e}\n{traceback.format_exc()}")
        raise


@app.post("/ask")
def ask(q: Question):
    """Accept a JSON payload {'question': '...'} and return an answer."""
    try:
        result = answer_query(q.question)
        # If result contains 'error', return 400
        if result.get("error"):
            raise HTTPException(status_code=400, detail=result.get("error"))
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unhandled exception in /ask")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


if __name__ == "__main__":
    # Allow running the API server directly: python rest_api.py
    import uvicorn

    uvicorn.run("rest_api:app", host="0.0.0.0", port=8000, reload=False)
