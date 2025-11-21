# Trust Agent REST API

This repository now includes a small REST API wrapper around the existing agent pipeline.

How to run (Windows PowerShell):

1. Create a virtual environment and activate it (optional but recommended):

   python -m venv .venv; .\.venv\Scripts\Activate.ps1

2. Install dependencies:

   python -m pip install --upgrade pip; python -m pip install -r requirements.txt

3. Ensure environment variables used by the agent are set (e.g., `ABACUS_API_KEY`).

4. Run the API server:

   python rest_api.py

   or with uvicorn directly:

   uvicorn rest_api:app --host 0.0.0.0 --port 8000

Endpoint:

- POST /ask
  - Request JSON: {"question": "Your question here"}
  - Response JSON: {"answer": "...", "sources": [...], "trusted_urls": [...]} or HTTP 400/500 with details

Notes:

- The REST API reuses the pipeline in `dynamic_rag_agent.py`. Running the endpoint may require the same heavy dependencies and services (SearxNG at localhost:8080, Abacus API key, crawl4ai, chromadb, etc.).
- If you want a lighter smoke-test harness (no external calls), ask and I can add a mock mode.
