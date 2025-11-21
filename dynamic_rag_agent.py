"""
Dynamic RAG Agent with Trust Filtering and Query Enrichment.

This module provides a RAG (Retrieval-Augmented Generation) system that:
- Enriches user queries using LLM-based extraction
- Filters search results through a trust scoring system
- Scrapes and indexes trusted sources
- Generates answers based on retrieved context

Migrated to use Abacus AI RouteLLM instead of Ollama.
"""

from llama_index.core import VectorStoreIndex, Settings, PromptTemplate, Document
# Embedding backend: prefer llama_index's HuggingFace wrapper when available.
_hf_err = None
try:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
except Exception as _hf_err:
    try:
        # Some versions expose a different import path
        from llama_index.embeddings import HuggingFaceEmbedding  # type: ignore
    except Exception as _inner_err:
        # Store the original error for the error message
        _original_hf_err = _hf_err
        # Provide a helpful error when the class is instantiated so the user
        # sees actionable guidance during container startup rather than an
        # import-time traceback.
        class HuggingFaceEmbedding:  # type: ignore
            def __init__(self, *args, **kwargs):
                err_msg = str(_original_hf_err) if _original_hf_err else str(_inner_err)
                raise ImportError(
                    "HuggingFaceEmbedding is not available in this environment. "
                    "Install 'llama-index-embeddings-huggingface' and 'sentence-transformers' packages. "
                    f"Original import error: {err_msg}"
                )
from llama_index.llms.openai import OpenAI  # RouteLLM uses OpenAI-compatible API
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.storage import StorageContext
from llama_index.core.query_engine import BaseQueryEngine
try:
    from llama_index.vector_stores.chroma import ChromaVectorStore
except Exception as _err:
    raise ImportError(
        "llama_index.vector_stores.chroma not available. "
        "Please ensure 'llama-index' is installed with Chroma support and rebuild the image. "
        "Example: pip install 'llama-index[chromadb]' or add an appropriate 'llama-index' release to requirements.txt. "
        f"Original error: {_err}"
    )
import chromadb
import requests
from crawl4ai import AsyncWebCrawler
import whois
import re
import json
import asyncio
import logging
from typing import List, Dict, Optional, Tuple, Any
from urllib.parse import urlparse
from datetime import datetime
import os

# Configure logging for better debugging and AI assistant understanding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration constants - makes code more maintainable and AI-friendly
class Config:
    """Application configuration constants."""
    # Abacus AI RouteLLM Configuration
    ABACUS_API_KEY: str = os.getenv("ABACUS_API_KEY", "")  # Set your API key here or via environment variable
    ABACUS_BASE_URL: str = os.getenv("ABACUS_BASE_URL", "https://routellm.abacus.ai/v1/")  # OpenAI compatible API base URL
    LLM_MODEL: str = "gpt-5-mini"  # You can use: gpt-4o, gpt-4o-mini, claude-3-5-sonnet, etc.
    LLM_TIMEOUT: float = 600.0  # seconds
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 512
    
    # Embedding Configuration
    EMBED_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Trust Scoring Configuration
    TRUST_THRESHOLD: float = 0.7
    MIN_TRUSTED_SOURCES: int = 2
    HTTPS_SCORE_BONUS: float = 0.1
    OFFICIAL_TLD_SCORE_BONUS: float = 0.2
    OFFICIAL_PATH_SCORE_BONUS: float = 0.2
    VENDOR_MATCH_SCORE_BONUS: float = 0.3
    NEUTRAL_SCORE: float = 0.5
    
    # Search Configuration
    # When running under docker-compose the searxng service is reachable at the service name `searxng`
    SEARXNG_URL: str = "http://searxng:8080/search"
    SEARXNG_TIMEOUT: int = 10  # seconds
    DEFAULT_NUM_RESULTS: int = 10
    # Try multiple engines in order, use first that returns results
    SEARCH_ENGINES: List[str] = ["duckduckgo", "google", "bing", "startpage"]
    
    # Scraping Configuration
    SCRAPE_TIMEOUT_PER_URL: int = 30  # seconds
    MAX_DOCUMENT_CHARS: int = 50000
    
    # Domain Info Configuration
    DOMAIN_CHECK_TIMEOUT: int = 5  # seconds
    WHOIS_TIMEOUT: int = 5  # seconds
    
    # Vector Store Configuration
    CHROMA_DB_PATH: str = "./temp_chroma"
    CHROMA_COLLECTION_NAME: str = "query_docs"
    SIMILARITY_TOP_K: int = 5  # Increased from 3 to get more retrieval candidates
    SIMILARITY_CUTOFF: float = 0.4  # Lowered from 0.5 to allow nodes with scores >= 0.4

# Configure local components using Config constants
Settings.embed_model = HuggingFaceEmbedding(model_name=Config.EMBED_MODEL)

# Configure Abacus AI RouteLLM
Settings.llm = OpenAI(
    model=Config.LLM_MODEL,
    api_key=Config.ABACUS_API_KEY,
    api_base=Config.ABACUS_BASE_URL,
    temperature=Config.LLM_TEMPERATURE,
    max_tokens=Config.LLM_MAX_TOKENS,
    timeout=Config.LLM_TIMEOUT
)

class TrustAgent:
    """
    Evaluates URL trustworthiness using heuristic rules and LLM-based scoring.
    
    Uses a two-stage approach:
    1. Fast heuristic scoring (rule-based, no LLM)
    2. LLM-based scoring for ambiguous cases
    
    Attributes:
        llm: The language model instance for LLM-based trust evaluation
        trust_prompt: Template for LLM-based trust evaluation
    """
    
    def __init__(self, llm: Any) -> None:
        """
        Initialize TrustAgent with an LLM instance.
        
        Args:
            llm: Language model for scoring URLs that don't pass heuristic checks
        """
        self.llm = llm
        self.trust_prompt = PromptTemplate(
            "Score this URL for trustworthiness as an official source (0-1, >0.7=trusted). "
            "Criteria: Official TLD (.gov/.edu/.org), 'docs/' path, HTTPS, domain age >1yr, "
            "vendor match (e.g., aws.amazon.com for AWS). Query context: {query}. "
            "URL: {url}. Domain info: {domain_info}. Output JSON: {{'score': float, 'reason': str}}"
        )
    
    def get_domain_info(self, url: str) -> str:
        """
        Retrieve domain information including age, registrar, and HTTPS status.
        
        Args:
            url: The URL to check domain information for
        
        Returns:
            A formatted string containing domain age, registrar, and HTTPS status.
            Returns "Unknown domain info" if the check fails.
        """
        try:
            domain = re.sub(r"https?://(www\.)?", "", url).split('/')[0]
            w = whois.whois(domain)
            age = 0
            if w.creation_date:
                if isinstance(w.creation_date, list):
                    creation_year = w.creation_date[0].year if w.creation_date[0] else None
                else:
                    creation_year = w.creation_date.year if hasattr(w.creation_date, 'year') else None
                if creation_year:
                    age = datetime.now().year - int(creation_year)
            
            registrar = w.registrar if hasattr(w, 'registrar') and w.registrar else "Unknown"
            
            # Check HTTPS with timeout
            try:
                https_check = requests.head(
                    url, 
                    timeout=Config.DOMAIN_CHECK_TIMEOUT, 
                    allow_redirects=True
                )
                hsts = https_check.headers.get('Strict-Transport-Security', 'No')
            except (requests.RequestException, requests.Timeout) as e:
                logger.debug(f"HTTPS check failed for {url}: {e}")
                hsts = "Unknown"
            
            return f"Age: {age}yrs, Registrar: {registrar}, HTTPS: {hsts}"
        except Exception as e:
            logger.error(f"Domain info error for {url}: {e}")
            return "Unknown domain info"
    
    def heuristic_score(self, url: str, query: str) -> Dict[str, Any]:
        """
        Fast rule-based URL trust scoring without LLM.
        
        Scoring criteria:
        - HTTPS: +0.1
        - Official TLD (.gov, .edu, .org, .io): +0.2
        - Official paths (docs, guide, official): +0.2
        - Vendor match: +0.3
        
        Args:
            url: The URL to score
            query: The user query for vendor matching
        
        Returns:
            Dictionary with 'score' (float 0-1), 'reason' (str), and 'method' ('heuristic')
        """
        parsed = urlparse(url)
        score = Config.NEUTRAL_SCORE
        reason: List[str] = []
        
        # HTTPS check
        if parsed.scheme == 'https':
            score += Config.HTTPS_SCORE_BONUS
            reason.append("HTTPS")
        
        # Official TLDs
        official_tlds = ['.gov', '.edu', '.org', '.io']
        if any(tld in parsed.netloc.lower() for tld in official_tlds):
            score += Config.OFFICIAL_TLD_SCORE_BONUS
            reason.append("Official TLD")
        
        # Docs/official paths
        official_paths = ['docs', 'guide', 'official']
        if any(path in parsed.path.lower() for path in official_paths):
            score += Config.OFFICIAL_PATH_SCORE_BONUS
            reason.append("Official path")
        
        # Vendor match from query (extensible pattern)
        query_lower = query.lower()
        vendor_matches = [
            ('quarkus', 'quarkus.io'),
            ('aws', 'aws.amazon.com'),
            ('kubernetes', 'kubernetes.io'),
        ]
        for keyword, domain in vendor_matches:
            if keyword in query_lower and domain in url.lower():
                score += Config.VENDOR_MATCH_SCORE_BONUS
                reason.append("Query-vendor match")
                break
        
        return {
            'score': min(score, 1.0),
            'reason': '; '.join(reason),
            'method': 'heuristic'
        }
    
    def score_url(self, url: str, query: str) -> Dict[str, Any]:
        """
        Score URL trustworthiness using heuristic first, then LLM if needed.
        
        Args:
            url: The URL to score
            query: The user query for context
        
        Returns:
            Dictionary with 'score', 'reason', and 'method' keys
        """
        # Try heuristic first (fast, no LLM cost)
        heur = self.heuristic_score(url, query)
        if heur['score'] >= Config.TRUST_THRESHOLD:
            logger.debug(f"Heuristic trusted {url} (score: {heur['score']})")
            return heur
        
        # Fallback to LLM if needed (slower, but with timeout safety)
        logger.debug(f"Using LLM for {url}")
        domain_info = self.get_domain_info(url)
        try:
            short_prompt = PromptTemplate(
                "Score URL trust (0-1, >0.7=trusted). Query: {query}. URL: {url}. "
                "Output JSON only: {{'score': float, 'reason': str}}"
            )
            print(f"DEBUG: Calling OpenAI LLM for URL scoring: {url}")
            response = self.llm.complete(short_prompt.format(query=query, url=url))
            print(f"DEBUG: OpenAI LLM response for {url}: {response.text}")
            response_text = response.text.strip()
            
            # Extract JSON from response (handle cases where model adds extra text)
            json_data = self._extract_json_from_text(response_text)
            if json_data:
                return json_data
            return json.loads(response_text)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"LLM timeout/error for {url}: {e}. Falling back to heuristic.")
            return self.heuristic_score(url, query)  # Graceful fallback
    
    @staticmethod
    def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON object from text that may contain extra content.
        
        Args:
            text: Text that may contain a JSON object
        
        Returns:
            Parsed JSON dictionary or None if extraction fails
        """
        start_idx = text.find('{')
        if start_idx == -1:
            return None
        
        brace_count = 0
        for i in range(start_idx, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    json_str = text[start_idx:i+1]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        return None
        return None
    
    def filter_trusted(
        self, 
        urls: List[str], 
        query: str, 
        min_trusted: int = Config.MIN_TRUSTED_SOURCES
    ) -> List[str]:
        """
        Filter URLs based on trust scores.
        
        Args:
            urls: List of URLs to filter
            query: User query for context
            min_trusted: Minimum number of trusted sources required
        
        Returns:
            List of trusted URLs, or empty list if insufficient trusted sources
        """
        scores: List[Tuple[str, Dict[str, Any]]] = []
        for url in urls:
            try:
                score_data = self.score_url(url, query)
                scores.append((url, score_data))
            except Exception as e:
                logger.error(f"Error scoring {url}: {e}. Skipping.")
                continue
        
        trusted = [
            url for url, data in scores 
            if data['score'] > Config.TRUST_THRESHOLD
        ]
        
        if len(trusted) < min_trusted:
            logger.warning(
                f"Trust Agent: Only {len(trusted)} trusted sources; need {min_trusted}."
            )
            return []
        
        logger.info(f"Trust Agent: Filtered to {len(trusted)} trusted URLs.")
        for url, data in scores:
            if data['score'] <= Config.TRUST_THRESHOLD:
                logger.debug(
                    f"Rejected {url}: {data['reason']} (score: {data['score']})"
                )
        return trusted

class QueryExtractorAgent:
    """
    Extracts and enriches user queries using LLM-based analysis.
    
    Preserves topic names while adding search enhancements like site: operators
    and official documentation keywords.
    """
    
    def __init__(self, llm: Any) -> None:
        """
        Initialize QueryExtractorAgent with an LLM instance.
        
        Args:
            llm: Language model for query extraction and enrichment
        """
        self.llm = llm
        self.extract_prompt = PromptTemplate(
            "You are a search query understanding assistant.\n\n"
            "Task:\n"
            "Given the user query below, extract and infer:\n"
            "1. topic: The main topic/entity (e.g., 'Quarkus').\n"
            "2. intent: The primary intent/purpose (e.g., 'details', 'overview', 'tutorial', 'installation', 'comparison').\n"
            "3. refinements: A list of additional qualifiers or subtopics (e.g., 'architecture', 'performance comparison with Spring Boot').\n"
            "4. enriched_query: A concise, improved version of the query optimized for web search.\n\n"
            "Enriched query rules:\n"
            "- Keep it short and precise.\n"
            "- Add double quotes around multi‑word key phrases that should stay together.\n"
            "- Prefer reliable sources (e.g., append terms like 'official documentation', 'reference guide' when appropriate).\n"
            "- If the topic looks like a known framework/library/project with an official site (e.g., Quarkus → quarkus.io), you MAY add a suitable site: operator.\n"
            "- Do not invent a site: operator if you are not reasonably confident.\n"
            "- Do not change the original meaning or add unsupported claims.\n\n"
            "Output format (MUST be valid JSON, no comments, no trailing commas):\n"
            "{\n"
            "  \"topic\": string,\n"
            "  \"intent\": string,\n"
            "  \"refinements\": [string, ...],\n"
            "  \"enriched_query\": string\n"
            "}\n\n"
            "User query: \"{query}\"\n"
        )
    
    def extract_and_enrich(self, query: str) -> List[str]:
        """Extract context and return multiple focused search queries.
        
        Returns a list of 3-5 focused queries instead of one long query to improve search results.
        """
        try:
            # Enhanced prompt that generates multiple focused queries
            short_prompt = PromptTemplate(
                "You are a search query optimizer. Your task is to transform user queries into multiple focused search engine queries.\n\n"
                "RULES:\n"
                "1. PRESERVE all key entities, product names, and technical terms exactly as written\n"
                "2. GENERATE 3-5 separate, focused queries (not one long query)\n"
                "3. Each query should be concise (5-10 words max) and focus on a specific aspect\n"
                "4. Vary the queries to cover different angles: methods, algorithms, libraries, tutorials, documentation\n"
                "5. USE quotes for exact phrase matching when appropriate\n"
                "6. REMOVE filler words (e.g., 'Can I', 'How do I', 'What is', 'Tell me about')\n"
                "7. DO NOT add site: operators\n"
                "8. DO NOT create overly long queries with many keywords\n\n"
                "EXAMPLES:\n"
                "- Input: 'What are methods to detect QR code 2D?'\n"
                "  Output: {{\"topic\": \"QR code detection\", \"queries\": [\"QR code detection methods\", \"2D barcode detection algorithms\", \"QR code OpenCV ZXing\", \"QR code detection tutorial\"]}}\n"
                "- Input: 'How to deploy Spring Boot to AWS?'\n"
                "  Output: {{\"topic\": \"Spring Boot AWS\", \"queries\": [\"Spring Boot AWS deployment\", \"Spring Boot deploy EC2\", \"Spring Boot AWS tutorial\"]}}\n\n"
                "User Query: {query}\n\n"
                "Output ONLY valid JSON with double quotes, no markdown, no code blocks:\n"
                "{{\"topic\": \"main entity or technology\", \"queries\": [\"query1\", \"query2\", \"query3\"]}}"
            )
            print(f"DEBUG: Calling OpenAI LLM for query enrichment: {query}")
            response = self.llm.complete(short_prompt.format(query=query))
            print(f"DEBUG: OpenAI LLM response for query enrichment: {response.text}")
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
            
            queries = data.get('queries', [])
            topic = data.get('topic', '').lower()
            
            # Validate and ensure we have queries
            if not queries or not isinstance(queries, list):
                # Fallback: create a simple query from the original
                print(f"DEBUG: No queries list found, creating fallback query")
                queries = [query]
            
            # Ensure topic is present in at least the first query if provided
            if topic and queries:
                first_query_lower = queries[0].lower()
                if topic not in first_query_lower:
                    queries[0] = f"{topic} {queries[0]}"
            
            # Limit to 5 queries max
            queries = queries[:5]
            
            print(f"DEBUG: Extracted - Topic: {data.get('topic', 'unknown')}, Queries: {queries}")
            return queries
        except Exception as e:
            print(f"DEBUG: Extraction failed: {e}. Using original query as single query.")
            return [query]  # Fallback to single query as list

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

def search_searxng(queries, num_results_per_query=10):
    """Query local SearxNG for multiple queries and collect all results.
    
    Args:
        queries: A single query string or a list of query strings
        num_results_per_query: Maximum number of results to fetch per query (default: 10)
    
    Returns:
        List of unique search results (deduplicated by URL)
    """
    # Normalize input: convert single query to list
    if isinstance(queries, str):
        queries = [queries]
    
    # Use configured SEARXNG_URL so the app works both on localhost and inside docker-compose
    url = Config.SEARXNG_URL
    engines_to_try = Config.SEARCH_ENGINES
    all_results = []
    
    # Search for each query
    for query in queries:
        print(f"DEBUG: Searching for query: {query}")
        query_results_found = False
        
        # Try engines in order until we get results
        for engine in engines_to_try:
            params = {
                "q": query,
                "format": "json",
                "engines": engine,
                "categories": "general",
                "safesearch": "0"
            }
            print(f"DEBUG: Trying engine '{engine}' for query: {query}")
            
            try:
                response = requests.get(url, params=params, timeout=Config.SEARXNG_TIMEOUT)
                print(f"DEBUG: Response status for {engine}: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                    
                    # Check if we got any results
                    if results:
                        results = results[:num_results_per_query]
                        print(f"DEBUG: Got {len(results)} results from {engine} for query: {query}")
                        query_results_found = True
                        
                        for r in results:
                            try:
                                all_results.append({
                                    "title": r.get("title", "No title"),
                                    "url": r.get("url", ""),
                                    "content": r.get("content", ""),
                                    "query": query,  # Track which query found this result
                                    "engine": engine  # Track which engine found this result
                                })
                            except (KeyError, TypeError) as e:
                                print(f"DEBUG: Error processing result: {e}")
                                continue
                        
                        # If we got results from this engine, no need to try others for this query
                        break
                    else:
                        print(f"DEBUG: Engine '{engine}' returned 0 results for query: {query}")
                else:
                    print(f"DEBUG: Failed {engine} with status {response.status_code} for query: {query}")
            except requests.exceptions.RequestException as e:
                print(f"DEBUG: Request error for {engine} on query '{query}': {e}")
            except json.JSONDecodeError as e:
                print(f"DEBUG: JSON parse error for {engine} on query '{query}': {e}")
        
        if not query_results_found:
            print(f"DEBUG: WARNING: No results found for query '{query}' from any engine")
    
    # Deduplicate by URL (keep first occurrence)
    seen_urls = set()
    unique_results = []
    for r in all_results:
        url = r.get("url", "")
        if url and url not in seen_urls:
            unique_results.append(r)
            seen_urls.add(url)
    
    print(f"DEBUG: Total results collected: {len(all_results)}, Unique results after deduplication: {len(unique_results)}")
    return unique_results

def scrape_with_crawl4ai(urls: List[str], timeout_per_url: int = 30):
    """Scrape URLs with timeout protection."""
    import asyncio
    
    async def scrape_url(crawler, url):
        """Scrape a single URL with timeout."""
        try:
            result = await asyncio.wait_for(
                crawler.arun(url=url),
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
        docs = []
        # Use async context manager for proper initialization and cleanup
        async with AsyncWebCrawler() as crawler:
            for url in urls:
                result = await scrape_url(crawler, url)
                if result and hasattr(result, 'markdown') and result.markdown:
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
    # web_tool = FunctionTool.from_defaults(fn=lambda q: search_searxng(q), name="Web Search")
    # trust_tool = FunctionTool.from_defaults(fn=lambda u, q: trust_agent.filter_trusted([u], q), name="trust_filter")
    # agent = ReActAgent.from_tools([web_tool, trust_tool], llm=Settings.llm, verbose=True)
    # But for simplicity, inline below

    while True:
        user_query = input("Ask (or 'quit'): ")
        if user_query.lower() == 'quit':
            break
        # Use QueryExtractorAgent to extract and enrich the query into multiple focused queries
        enriched_queries = query_extractor.extract_and_enrich(user_query)

        # Step 1: Search with multiple queries (each returns up to 10 results)
        search_results = search_searxng(enriched_queries, num_results_per_query=10)

        if not search_results:
            print("No sources found.")
            continue

        # Step 2: Trust filter
        urls = [r.get("url", "") for r in search_results if r.get("url")]
        if not urls:
            print("No valid URLs found in search results.")
            continue
        trusted_urls = trust_agent.filter_trusted(urls, user_query)
        if not trusted_urls:
            print("Trust Agent: Insufficient reliable sources after filtering.")
            continue

        # Step 3: Scrape trusted only
        scraped_docs = scrape_with_crawl4ai(trusted_urls)
        print(f"DEBUG: Number of scraped docs: {len(scraped_docs)}")
        for i, doc in enumerate(scraped_docs):
            print(f"DEBUG: Doc {i}: URL={doc.get('url', 'unknown')}, Text length={len(doc.get('text', ''))}")
        if not scraped_docs:
            print("Scraping failed.")
            continue

        # Step 4: Create documents from scraped content
        documents = []
        for doc in scraped_docs:
            try:
                documents.append(Document(
                    text=doc["text"],
                    metadata={"url": doc["url"]}
                ))
            except Exception as e:
                print(f"DEBUG: Error creating document for {doc.get('url', 'unknown')}: {e}")
                continue
        
        print(f"DEBUG: Number of documents created: {len(documents)}")
        if not documents:
            print("No valid documents created from scraped content.")
            continue
        
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

        # Debug retrieval before querying
        retriever = index.as_retriever(similarity_top_k=Config.SIMILARITY_TOP_K)
        nodes = retriever.retrieve(user_query)
        print(f"DEBUG: Retrieved {len(nodes)} nodes before postprocessing")
        for i, node in enumerate(nodes):
            print(f"DEBUG: Node {i} score: {node.score:.3f}, URL: {node.metadata.get('url', 'unknown')}, text preview: {node.text[:200]}...")
        
        postprocessor = SimilarityPostprocessor(similarity_cutoff=Config.SIMILARITY_CUTOFF)
        from llama_index.core.schema import QueryBundle
        query_bundle = QueryBundle(query_str=user_query)
        filtered_nodes = postprocessor.postprocess_nodes(nodes, query_bundle=query_bundle)
        print(f"DEBUG: After postprocessing with cutoff {Config.SIMILARITY_CUTOFF}, {len(filtered_nodes)} nodes remain")

        # Step 5: Retrieve and query
        # Use index.as_query_engine - it creates its own retriever internally
        # Pass similarity_top_k instead of retriever to avoid conflict
        query_engine = index.as_query_engine(
            similarity_top_k=Config.SIMILARITY_TOP_K,
            node_postprocessors=[postprocessor],
            system_prompt=system_prompt,
            verbose=True  # Enable verbose logging for debugging retrieval and LLM calls
        )
        try:
            response = query_engine.query(user_query)
            print(f"DEBUG: Query response object: {response}")
            print(f"DEBUG: Response text: '{response.response}'")
            print(f"Answer: {response}\nSources: {response.get_formatted_sources()}")
        except Exception as e:
            print(f"ERROR: Query processing failed: {e}")
            print("This might be due to timeout or context size. Try a simpler query or check model performance.")

if __name__ == "__main__":
    # Check if API key is set
    if not Config.ABACUS_API_KEY:
        print("ERROR: ABACUS_API_KEY not set. Please set it as an environment variable or in the Config class.")
        print("You can get your API key from: https://abacus.ai/app/route-llm-apis")
        exit(1)
    
    run_agent()