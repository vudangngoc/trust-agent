# **Trust Agent - RAG Agent with Trust Filtering**  
**Accurate • Trusted Sources Only • No Hallucinations**

This guide helps you set up and run a **RAG (Retrieval-Augmented Generation) agent** that:
- Searches **only trusted, official documentation**
- Uses **Abacus AI RouteLLM** for LLM inference
- **Never hallucinates** (refuses if no reliable source found)
- Filters results through a **trust scoring system**
- Provides answers via **REST API**

---

## Overview

| Component | Tool | Why |
|--------|------|-----|
| **LLM** | `Abacus AI RouteLLM` | OpenAI-compatible API, multiple model support |
| **Search** | `SearxNG` (Docker) | Private, multi-engine search (DuckDuckGo, Google, Bing, Startpage) |
| **Scraping** | `crawl4ai` + `Playwright` | JS-heavy docs support |
| **RAG** | `LlamaIndex` | Grounded answers with vector search |
| **Trust & Enrichment** | Heuristic + LLM-based | Filters to official sources only |

---

## Prerequisites

| Tool | Install |
|------|--------|
| **Docker Desktop** | [docker.com](https://www.docker.com/products/docker-desktop/) |
| **Git** | [git-scm.com](https://git-scm.com/downloads) |
| **Abacus AI API Key** | [abacus.ai/app/route-llm-apis](https://abacus.ai/app/route-llm-apis) |

> **Note**: Works on macOS, Windows, and Linux with Docker Desktop.

---

## Quick Start

### 1. Clone the Repository
```bash
git clone <repository-url>
cd trust-agent
```

### 2. Configure Environment Variables

Create a `.env` file in the project root:

```bash
# Abacus AI RouteLLM Configuration
ABACUS_API_KEY=your_api_key_here
ABACUS_BASE_URL=https://routellm.abacus.ai/v1/
```

> **Get your API key**: Visit [abacus.ai/app/route-llm-apis](https://abacus.ai/app/route-llm-apis) to obtain your `ABACUS_API_KEY`.

### 3. Build and Start Services

```bash
docker-compose up -d --build
```

This will start:
- **Trust Agent API** on port `8000`
- **SearxNG** search engine on port `8080`
- **Redis** for SearxNG caching

### 4. Verify Services are Running

Check that all containers are up:
```bash
docker-compose ps
```

You should see:
- `trust-agent-app` (running)
- `searxng` (running)
- `searxng-redis` (running)

Test SearxNG: Open http://localhost:8080 in your browser and perform a test search.

---

## Usage

### REST API

The Trust Agent exposes a REST API at `http://localhost:8000`.

#### Ask a Question

```bash
curl --location 'http://localhost:8000/ask' \
--header 'Content-Type: application/json' \
--data '{"question":"What are methods to detect QR code 2D?"}'
```

**Example Request:**
```json
{
  "question": "What are methods to detect QR code 2D?"
}
```

**Example Response:**
```json
{
  "answer": "QR code detection methods include...",
  "sources": ["[Source: https://example.com/qr-detection]"],
  "trusted_urls": [
    "https://docs.opencv.org/qr-detection",
    "https://github.com/zxing/zxing"
  ]
}
```

#### Response Format

- **`answer`**: The generated answer based on trusted sources
- **`sources`**: List of formatted source citations
- **`trusted_urls`**: URLs of trusted sources used

#### Error Responses

- **400 Bad Request**: Invalid input or insufficient trusted sources found
- **500 Internal Server Error**: Processing error (check logs)

---

## Architecture

### How It Works

1. **Query Enrichment**: User query is enriched into multiple focused search queries using LLM
2. **Multi-Engine Search**: Searches across multiple engines (DuckDuckGo, Google, Bing, Startpage) with automatic fallback
3. **Trust Filtering**: URLs are scored for trustworthiness using heuristics and LLM-based evaluation
4. **Content Scraping**: Trusted URLs are scraped using Playwright
5. **Vector Indexing**: Scraped content is indexed in ChromaDB
6. **Answer Generation**: LLM generates answer based on retrieved context with source citations

### Trust Scoring Criteria

- **HTTPS**: +0.1
- **Official TLD** (.gov, .edu, .org, .io): +0.2
- **Official paths** (docs/, guide/, official/): +0.2
- **Vendor match** (e.g., aws.amazon.com for AWS queries): +0.3
- **Domain age** and **registrar info** (via LLM evaluation)

Minimum trust threshold: **0.7**

---

## Configuration

### Environment Variables

Edit `.env` file:

```bash
# Abacus AI RouteLLM or any OpenAI competitive API
ABACUS_API_KEY=your_api_key_here
ABACUS_BASE_URL=https://routellm.abacus.ai/v1/

# Optional: Override defaults in dynamic_rag_agent.py
# LLM_MODEL=gpt-5-mini
# TRUST_THRESHOLD=0.7
```

### SearxNG Configuration

Edit `searxng/settings.yml` to enable/disable search engines:

```yaml
engines:
  - name: duckduckgo
    disabled: false
  - name: google
    disabled: false
  - name: bing
    disabled: false
  - name: startpage
    disabled: false
```

### Application Configuration

Edit `dynamic_rag_agent.py` `Config` class to adjust:
- `LLM_MODEL`: Model to use (gpt-5-mini, gpt-4o, claude-3-5-sonnet, etc.)
- `TRUST_THRESHOLD`: Minimum trust score (default: 0.7)
- `SEARCH_ENGINES`: List of engines to try in order
- `SIMILARITY_TOP_K`: Number of retrieved chunks (default: 5)
- `SIMILARITY_CUTOFF`: Minimum similarity score (default: 0.4)

---

## Development

### Rebuild After Code Changes

```bash
docker-compose up -d --build
```

### View Logs

```bash
# All services
docker-compose logs -f

# Trust Agent only
docker-compose logs -f app

# SearxNG only
docker-compose logs -f searxng
```

### Stop Services

```bash
docker-compose down
```

### Clean Start (Remove Volumes)

```bash
docker-compose down -v
```

---

## Troubleshooting

| Error | Cause | Fix |
|------|-------|-----|
| `ABACUS_API_KEY not set` | Missing API key | Add `ABACUS_API_KEY` to `.env` file |
| `Connection refused` on port 8000 | Container not running | `docker-compose up -d` |
| `0 results from duckduckgo` | Engine issue | System auto-falls back to Google/Bing/Startpage |
| `playwright: Executable doesn't exist` | Browser not installed | Rebuild image: `docker-compose build` |
| `SQLite version error` | Old SQLite | Ensure using `python:3.11-slim-bookworm` base image |
| `ImportError: llama_index.vector_stores.chroma` | Missing package | Check `requirements.txt` includes `llama-index-vector-stores-chroma` |
| `SearxNG 403 on JSON` | JSON format not enabled | Ensure `searxng/settings.yml` has `formats: [html, json]` |
| Port 8000 or 8080 in use | Port conflict | Change ports in `docker-compose.yml` |

### Check Container Status

```bash
docker-compose ps
docker-compose logs app
```

### Test SearxNG Directly

```bash
curl "http://localhost:8080/search?q=test&format=json"
```

### Verify Environment Variables

```bash
docker-compose exec app env | grep ABACUS
```

---

## Project Structure

```
trust-agent/
├── docker-compose.yml          # Docker services configuration
├── Dockerfile                  # Trust Agent container image
├── .env                        # Environment variables (create this)
├── requirements.txt            # Python dependencies
├── dynamic_rag_agent.py       # Core RAG agent logic
├── rest_api.py                # FastAPI REST endpoint
├── searxng/
│   ├── settings.yml           # SearxNG configuration
│   └── data/                  # SearxNG data directory
└── temp_chroma/               # ChromaDB vector store (created at runtime)
```

---

## API Documentation

For detailed API documentation, see [API_README.md](API_README.md).

---

## License

See [LICENSE](LICENSE) file for details.
