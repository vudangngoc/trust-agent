import pandas as pd
from llama_index.core import VectorStoreIndex, Settings, PromptTemplate
from llama_index.core.tools import FunctionTool  # Only if you plan to use tools later
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.storage import StorageContext
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.agent import ReActAgent
import chromadb
import requests
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler
import whois
import re
from typing import List, Dict
from llama_index.vector_stores.chroma import ChromaVectorStore
import json
from urllib.parse import urlparse

# Configure local components
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
# Increased timeout and added max_tokens to prevent long generations
Settings.llm = Ollama(
    model="phi3:3.8b", 
    request_timeout=600.0,  # Increased from 120 to 600 seconds
    temperature=0.1,  # Lower temperature for faster, more deterministic responses
    max_tokens=512  # Limit response length to prevent very long generations
)

# Trust Agent: Dynamic evaluator
class TrustAgent:
    def __init__(self, llm):
        self.llm = llm
        self.trust_prompt = PromptTemplate(
            "Score this URL for trustworthiness as an official source (0-1, >0.7=trusted). "
            "Criteria: Official TLD (.gov/.edu/.org), 'docs/' path, HTTPS, domain age >1yr, "
            "vendor match (e.g., aws.amazon.com for AWS). Query context: {query}. "
            "URL: {url}. Domain info: {domain_info}. Output JSON: {{'score': float, 'reason': str}}"
        )
    
    def get_domain_info(self, url: str) -> str:
        """Quick whois/metadata check (with timeout)."""
        try:
            domain = re.sub(r"https?://(www\.)?", "", url).split('/')[0]
            w = whois.whois(domain)
            age = (2025 - int(w.creation_date[0].year)) if w.creation_date else 0  # Approx for 2025
            return f"Age: {age}yrs, Registrar: {w.registrar}, HTTPS: {requests.head(url, timeout=5).headers.get('Strict-Transport-Security', 'No')}"
        except:
            return "Unknown domain info"
    
    def heuristic_score(self, url: str, query: str) -> Dict:
        """Fast rule-based scoring (no LLM)."""
        parsed = urlparse(url)
        score = 0.5  # Neutral start
        reason = []
        
        # HTTPS
        if parsed.scheme == 'https':
            score += 0.1
            reason.append("HTTPS")
        
        # Official TLDs
        official_tlds = ['.gov', '.edu', '.org', '.io']  # Add more as needed
        if any(tld in parsed.netloc.lower() for tld in official_tlds):
            score += 0.2
            reason.append("Official TLD")
        
        # Docs/official paths
        if 'docs' in parsed.path.lower() or 'guide' in parsed.path.lower() or 'official' in parsed.path.lower():
            score += 0.2
            reason.append("Official path")
        
        # Vendor match from query (simple keyword)
        query_lower = query.lower()
        if 'quarkus' in query_lower and 'quarkus.io' in url.lower():
            score += 0.3
            reason.append("Query-vendor match")
        # Extend for other topics: elif 'aws' in query_lower and 'aws.amazon.com' in url.lower(): score += 0.3
        
        return {'score': min(score, 1.0), 'reason': '; '.join(reason), 'method': 'heuristic'}
    
    def score_url(self, url: str, query: str) -> Dict:
        # Try heuristic first (fast)
        heur = self.heuristic_score(url, query)
        if heur['score'] >= 0.7:
            print(f"DEBUG: Heuristic trusted {url} (score: {heur['score']})")
            return heur
        
        # Fallback to LLM if needed (slower, but with timeout safety)
        print(f"DEBUG: Using LLM for {url}")
        domain_info = self.get_domain_info(url)
        try:
            # Use a shorter, more focused prompt to reduce generation time
            short_prompt = PromptTemplate(
                "Score URL trust (0-1, >0.7=trusted). Query: {query}. URL: {url}. "
                "Output JSON only: {{'score': float, 'reason': str}}"
            )
            response = self.llm.complete(short_prompt.format(query=query, url=url))
            import json
            # Try to extract JSON from response (handle cases where model adds extra text)
            response_text = response.text.strip()
            # Find JSON object by finding first { and matching closing }
            start_idx = response_text.find('{')
            if start_idx != -1:
                brace_count = 0
                for i in range(start_idx, len(response_text)):
                    if response_text[i] == '{':
                        brace_count += 1
                    elif response_text[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_str = response_text[start_idx:i+1]
                            return json.loads(json_str)
            # Try parsing the whole response
            return json.loads(response_text)
        except Exception as e:
            print(f"DEBUG: LLM timeout/error for {url}: {e}. Falling back to heuristic.")
            return self.heuristic_score(url, query)  # Graceful fallback
    
    def filter_trusted(self, urls: List[str], query: str, min_trusted: int = 2) -> List[str]:
        # Process URLs with timeout protection per URL
        scores = []
        for url in urls:
            try:
                score_data = self.score_url(url, query)
                scores.append((url, score_data))
            except Exception as e:
                print(f"DEBUG: Error scoring {url}: {e}. Skipping.")
                continue
        
        trusted = [url for url, data in scores if data['score'] > 0.7]
        if len(trusted) < min_trusted:
            print(f"Trust Agent: Only {len(trusted)} trusted sources; need more.")
            return []
        print(f"Trust Agent: Filtered to {len(trusted)} trusted URLs.")
        for url, data in scores:
            if data['score'] <= 0.7:
                print(f"Rejected {url}: {data['reason']} (score: {data['score']})")
        return trusted

class QueryExtractorAgent:
    def __init__(self, llm):
        self.llm = llm
        self.extract_prompt = PromptTemplate(
            "Analyze this user query and extract: "
            "1. Main topic/entity (e.g., 'Quarkus'). "
            "2. Intent (e.g., 'details', 'overview', 'tutorial'). "
            "3. Refinements (e.g., 'architecture', 'comparison'). "
            "Output JSON: {{'topic': str, 'intent': str, 'refinements': list[str], 'enriched_query': str}}. "
            "Enrich the query for precise search: Add quotes for phrases, 'official docs' for reliability, site: operators if topic is a known project (e.g., site:quarkus.io). "
            "Keep concise. Query: {query}"
        )
    
    def extract_and_enrich(self, query: str) -> str:
        """Extract context and return enriched search query."""
        try:
            # Use shorter prompt for faster generation
            short_prompt = PromptTemplate(
                "Extract topic and enrich query. Query: {query}. "
                "Output JSON: {{'topic': str, 'enriched_query': str}}"
            )
            response = self.llm.complete(short_prompt.format(query=query))
            import json
            response_text = response.text.strip()
            # Find JSON object by finding first { and matching closing }
            start_idx = response_text.find('{')
            if start_idx != -1:
                brace_count = 0
                for i in range(start_idx, len(response_text)):
                    if response_text[i] == '{':
                        brace_count += 1
                    elif response_text[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_str = response_text[start_idx:i+1]
                            data = json.loads(json_str)
                            break
                else:
                    data = json.loads(response_text)
            else:
                data = json.loads(response_text)
            print(f"DEBUG: Extracted - Topic: {data.get('topic', 'unknown')}, Enriched: {data.get('enriched_query', query)}")
            return data.get('enriched_query', query)
        except Exception as e:
            print(f"DEBUG: Extraction failed: {e}. Using original query.")
            return query  # Fallback to raw query

class QueryEnricher:
    def __init__(self):
        # Known official domains for common PoC topics
        self.official_sites = {
            "quarkus": ["quarkus.io", "redhat.com"],
            "aws": ["docs.aws.amazon.com", "aws.amazon.com"],
            "kubernetes": ["kubernetes.io"],
            "spring": ["spring.io", "docs.spring.io"],
            "java": ["docs.oracle.com/java"],
            "python": ["docs.python.org"],
            "docker": ["docs.docker.com"],
            "primary school": ["khanacademy.org", "education.com", "britannica.com"],
            "exercise": ["education.com", "superteacherworksheets.com"],
            # Add more as your PoCs grow
        }

    def enrich(self, query: str) -> str:
        query_lower = query.lower().strip()
        enriched = query

        # 1. Extract potential topic (simple noun/phrase)
        topic = self._extract_topic(query_lower)

        # 2. Add quotes for exact match if short
        if len(query.split()) <= 5:
            enriched = f'"{query}"'

        # 3. Add official site: if topic known
        if topic in self.official_sites:
            sites = " OR ".join([f"site:{s}" for s in self.official_sites[topic]])
            enriched = f"{enriched} {sites}"
            print(f"DEBUG: Enriched with official sites: {sites}")
        else:
            # Fallback: add "official documentation" for tech queries
            tech_keywords = ["framework", "library", "tool", "api", "language", "platform"]
            if any(k in query_lower for k in tech_keywords):
                enriched = f'{enriched} "official documentation"'

        # 4. Add file type for docs
        if any(w in query_lower for w in ["guide", "tutorial", "docs", "reference"]):
            enriched = f"{enriched} filetype:pdf OR filetype:md"

        print(f"DEBUG: Enriched query: {enriched}")
        return enriched

    def _extract_topic(self, query: str) -> str:
        # Simple: take first noun after common verbs
        query = re.sub(r"^(what is|tell me|explain|details about|how to|design)\s+", "", query, flags=re.I)
        query = re.sub(r"\s+(for|in|with|using).*$", "", query, flags=re.I)
        return query.strip().split()[0] if query else ""

# Build search query (now topic-agnostic; trust agent handles filtering)
def build_search_query(user_query):
    return user_query  # Broad search; trust agent narrows

def search_searxng(query, num_results=5):
    """Query local SearxNG for restricted results with debugging."""
    url = "http://localhost:8080/search"
    # Try multiple engines for robustness
    engines_to_try = ["google"]
    all_results = []
    
    for engine in engines_to_try:
        params = {
            "q": query,
            "format": "json",
            "engines": engine,
            "categories": "general",
            "safesearch": "0"
        }
        print(f"DEBUG: Trying engine '{engine}' for query: {query}")  # Debug log
        
        try:
            response = requests.get(url, params=params, timeout=10)  # Increased timeout
            print(f"DEBUG: Response status for {engine}: {response.status_code}")  # Debug
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])[:num_results]
                print(f"DEBUG: Got {len(results)} results from {engine}")  # Debug
                all_results.extend([{"title": r["title"], "url": r["url"], "content": r.get("content", "")} for r in results])
            else:
                print(f"DEBUG: Failed {engine} with status {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"DEBUG: Request error for {engine}: {e}")
        except json.JSONDecodeError as e:
            print(f"DEBUG: JSON parse error for {engine}: {e}")
    
    # Dedupe and limit total results
    seen_urls = set()
    unique_results = []
    for r in all_results:
        if r["url"] not in seen_urls:
            unique_results.append(r)
            seen_urls.add(r["url"])
        if len(unique_results) >= num_results:
            break
    
    print(f"DEBUG: Final unique results: {len(unique_results)}")  # Debug
    return unique_results

def scrape_with_crawl4ai(urls: List[str], timeout_per_url: int = 30):
    """Scrape URLs with timeout protection."""
    import asyncio
    
    async def scrape_url(crawler, url):
        """Scrape a single URL with timeout."""
        try:
            result = await asyncio.wait_for(
                crawler.crawl(url=url, output_format="markdown"),
                timeout=timeout_per_url
            )
            return result
        except asyncio.TimeoutError:
            print(f"DEBUG: Scraping timeout for {url} after {timeout_per_url}s")
            return None
        except Exception as e:
            print(f"DEBUG: Scraping error for {url}: {e}")
            return None
    
    async def scrape_all():
        """Scrape all URLs."""
        async with AsyncWebCrawler() as crawler:
            docs = []
            for url in urls:
                result = await scrape_url(crawler, url)
                if result and result.markdown:
                    # Limit document size to prevent memory issues
                    max_chars = 50000  # Limit to ~50k chars per document
                    text = result.markdown[:max_chars] if len(result.markdown) > max_chars else result.markdown
                    docs.append({"text": text, "url": url})
                    print(f"DEBUG: Scraped {url} ({len(text)} chars)")
            return docs
    
    # Run async scraping with proper event loop handling
    try:
        # Check if we're in an async context (get_running_loop raises RuntimeError if no loop)
        asyncio.get_running_loop()
        # If we get here, we're in an async context - this shouldn't happen in sync code
        print("WARNING: scrape_with_crawl4ai called from async context. Results may be incomplete.")
        docs = []
    except RuntimeError:
        # No running loop, safe to use asyncio.run
        try:
            docs = asyncio.run(scrape_all())
        except Exception as e:
            print(f"DEBUG: Error in scraping: {e}")
            docs = []
    
    return docs

# Main RAG prompt (unchanged)
system_prompt = PromptTemplate(
    "You are a precise assistant using only official docs. Answer based on {context_str}. "
    "If similarity <0.7 or no relevant info, say: 'Insufficient reliable sources found for this query.' "
    "Cite [Source: URL] for each fact."
)

# Interactive agent loop
def run_agent():
    db = chromadb.PersistentClient(path="./temp_chroma")
    chroma_collection = db.get_or_create_collection("query_docs")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Init trust agent and query extractor agent
    trust_agent = TrustAgent(Settings.llm)
    query_extractor = QueryExtractorAgent(Settings.llm)
    
    # Optional: Wrap in ReActAgent for more complex flows
    # from llama_index.core.tools import QueryEngineTool
    # web_tool = FunctionTool.from_defaults(fn=lambda q: search_searxng(q), name="web_search")
    # trust_tool = FunctionTool.from_defaults(fn=lambda u, q: trust_agent.filter_trusted([u], q), name="trust_filter")
    # agent = ReActAgent.from_tools([web_tool, trust_tool], llm=Settings.llm, verbose=True)
    # But for simplicity, inline below

    while True:
        user_query = input("Ask (or 'quit'): ")
        if user_query.lower() == 'quit':
            break
        # Use QueryExtractorAgent to extract and enrich the query
        enriched_query = query_extractor.extract_and_enrich(user_query)

        # Step 1: Broad search (now with enriched query)
        search_q = build_search_query(enriched_query)  # Pass enriched
        search_results = search_searxng(search_q)

        if not search_results:
            print("No sources found.")
            continue

        # Step 2: Trust filter
        urls = [r["url"] for r in search_results]
        trusted_urls = trust_agent.filter_trusted(urls, user_query)
        if not trusted_urls:
            print("Trust Agent: Insufficient reliable sources after filtering.")
            continue

        # Step 3: Scrape trusted only
        scraped_docs = scrape_with_crawl4ai(trusted_urls)
        if not scraped_docs:
            print("Scraping failed.")
            continue

        # Step 4: Temp index
        temp_docs = [SimpleDirectoryReader(input_files=[{"file_path": None, "text": d["text"], "metadata": {"url": d["url"]}}]) for d in scraped_docs]
        flat_docs = []
        for reader in temp_docs:
            flat_docs.extend(reader.load_data())
        index = VectorStoreIndex.from_documents(flat_docs, storage_context=storage_context)

        # Step 5: Retrieve and query
        retriever = VectorIndexRetriever(index=index, similarity_top_k=3)
        postprocessor = SimilarityPostprocessor(similarity_cutoff=0.7)
        query_engine = CustomQueryEngine.from_defaults(
            retriever=retriever, node_postprocessors=[postprocessor], system_prompt=system_prompt
        )
        try:
            response = query_engine.query(user_query)
            print(f"Answer: {response}\nSources: {response.get_formatted_sources()}")
        except Exception as e:
            print(f"ERROR: Query processing failed: {e}")
            print("This might be due to timeout or context size. Try a simpler query or check model performance.")

if __name__ == "__main__":
    run_agent()
