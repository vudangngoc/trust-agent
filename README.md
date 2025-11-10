# **Local RAG Agent for PoCs – macOS Setup Guide**  
**Accurate • Local • No Hallucinations • Official Docs Only**

This guide helps **Solution Architects** build a **fully local, open-source RAG agent** on **macOS** that:
- Searches **only official documentation**
- Uses **local LLM (Ollama)**
- **Never hallucinates** (refuses if no source)
- Works **entirely offline** after setup
- Handles **niche PoC topics** with precision

---

## Overview

| Component | Tool | Why |
|--------|------|-----|
| **LLM** | `ollama` + `phi3:3.8b` | Fast, local, 20+ tokens/sec |
| **Search** | `SearxNG` (Docker) | Private, domain-restricted |
| **Scraping** | `crawl4ai` + `Playwright` | JS-heavy docs |
| **RAG** | `LlamaIndex` | Grounded answers |
| **Trust & Enrichment** | Heuristic + Fallback | No CSV, adaptive |

---

## Prerequisites (macOS)

| Tool | Install |
|------|--------|
| **Python 3.12** | [python.org](https://www.python.org/downloads/) or `brew install python@3.12` |
| **Git** | Pre-installed, or [git-scm.com](https://git-scm.com/download/mac) |
| **Docker Desktop** | [docker.com](https://www.docker.com/products/docker-desktop/) |
| **VS Code** (Optional) | [code.visualstudio.com](https://code.visualstudio.com/) |

> **Docker Tip**: Docker Desktop for Mac works out of the box.

---

## Step-by-Step Setup

### 1. Create Project Folder
```bash
mkdir ~/rag-poc
cd ~/rag-poc
```

---

### 2. Create Virtual Environment
```bash
python3 -m venv rag-agent-env
source rag-agent-env/bin/activate
```

> You’ll see `(rag-agent-env)` in your prompt.

---

### 3. Install Python Packages
```bash
pip install --upgrade pip
pip install llama-index llama-index-llms-ollama llama-index-embeddings-huggingface \
            llama-index-vector-stores-chroma chromadb crawl4ai playwright \
            requests beautifulsoup4 pandas pypdf python-whois duckduckgo-search
```

> **Playwright Browsers** (required for scraping):
```bash
playwright install chromium
```

---

### 4. Install & Start Ollama
1. Download: [ollama.com](https://ollama.com/download) → Choose macOS installer
2. Install → Open **Terminal**
3. Pull model:
```bash
ollama pull phi3:3.8b
```
4. Start server:
```bash
ollama serve
```
> Keep this terminal **open**.

---

### 5. Start SearxNG (Local Search Engine)

#### Option A: Docker Compose (Recommended)
```bash
mkdir searxng && cd searxng
```
Create `docker-compose.yml`:
```yaml
version: '3'
services:
  caddy:
    image: caddy:2
    ports:
      - "8080:8080"
    volumes:
      - ./caddy/Caddyfile:/etc/caddy/Caddyfile
      - caddy_data:/data
    depends_on:
      - searxng

  searxng:
    image: searxng/searxng:latest
    volumes:
      - ./searxng:/etc/searxng
    environment:
      - AUTOSAVE=1
    depends_on:
      - redis

  redis:
    image: redis:7-alpine
    command: redis-server --save 60 1 --loglevel warning
    volumes:
      - redis_data:/data

volumes:
  caddy_data:
  redis_data:
```

Create folders:
```bash
mkdir -p searxng/caddy
touch searxng/settings.yml
```

Edit `searxng/settings.yml`:
```yaml
server:
  secret_key: your-super-secret-key-1234567890  # CHANGE THIS!
  bind_address: "0.0.0.0"
  port: 8080

search:
  formats:
    - html
    - json  # REQUIRED

engines:
  - name: duckduckgo
    disabled: false
  - name: google
    disabled: false
  - name: bing
    disabled: false
```

Start:
```bash
docker compose up -d
```

Test: http://localhost:8080 → Search "test" → Should work.

---

### 6. Save Agent Code

Create `dynamic_rag_agent.py` in `~/rag-poc`:

```python
# [FULL CODE BELOW - COPY & SAVE]
```

> **See full code at the end of this README**

---

### 7. Run the Agent
```bash
cd ~/rag-poc
source rag-agent-env/bin/activate
python3 dynamic_rag_agent.py
```

---

## Test Queries

| Query | Expected |
|------|----------|
| `tell me about Quarkus` | Official docs from `quarkus.io` |
| `generate math exercise for grade 3` | From `khanacademy.org` |
| `how to use AWS Lambda` | `docs.aws.amazon.com` |

---

## Troubleshooting (macOS)

| Error | Cause | Fix |
|------|-------|-----|
| `source: command not found` | Using wrong shell | Use `bash` or `zsh` (default on macOS) |
| `rag-agent-env/bin/activate: No such file` | Venv not created | `python3 -m venv rag-agent-env` |
| `playwright: Executable doesn't exist` | Browsers not installed | `playwright install chromium` |
| `Ollama timeout` | Model too slow | Use `phi3:3.8b` or increase `request_timeout=180` |
| `SearxNG 403 on JSON` | `json` not in `formats` | Add `- json` in `settings.yml` |
| `secret_key` error | Default key | Change to random string |
| `Docker port 8080 in use` | Another app | Change to `8081:8080` in `docker-compose.yml` |
| `python: command not found` | Python not in PATH | Use `python3` instead of `python` |
| `crawl4ai hangs` | JS site | `playwright install chromium` + `wait_for` in code |
| `Permission denied` | File permissions | `chmod +x` or check file ownership |

---