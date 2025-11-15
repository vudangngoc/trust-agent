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
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI  # RouteLLM uses OpenAI-compatible API
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.storage import StorageContext
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.vector_stores.chroma import ChromaVectorStore
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
    ABACUS_BASE_URL: str = "https://routellm.abacus.ai/v1/"  # RouteLLM endpoint
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
    SEARXNG_URL: str = "http://localhost:8080/search"
    SEARXNG_TIMEOUT: int = 10  # seconds
    DEFAULT_NUM_RESULTS: int = 10
    SEARCH_ENGINES: List[str] = ["google"]
    
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
            # Enhanced prompt that preserves topic names and adds search enhancements
            short_prompt = PromptTemplate(
                "You are a search query optimizer. Your task is to transform user queries into effective search engine queries.\n\n"
                "RULES:\n"
                "1. PRESERVE all key entities, product names, and technical terms exactly as written (e.g., 'JavaCV', 'GraalVM', 'Quarkus')\n"
                "2. EXTRACT the core question or intent (e.g., 'compatibility', 'tutorial', 'comparison', 'error fix')\n"
                "3. ADD relevant technical keywords that would appear in official documentation\n"
                "4. USE quotes for exact phrase matching when appropriate (e.g., \"error message\" or \"specific feature\")\n"
                "5. ADD context terms like 'official documentation', 'guide', 'reference' for authoritative sources\n"
                "6. REMOVE filler words (e.g., 'Can I', 'How do I', 'What is', 'Tell me about')\n"
                "7. DO NOT add site: operators\n"
                "8. DO NOT replace technical terms with generic descriptions\n\n"
                "EXAMPLES:\n"
                "- Input: 'Can I use JavaCV on GraalVM?' → Output: 'JavaCV GraalVM compatibility official documentation'\n"
                "- Input: 'What is Quarkus?' → Output: 'Quarkus framework official guide introduction'\n"
                "- Input: 'How to deploy Spring Boot to AWS?' → Output: 'Spring Boot AWS deployment guide official documentation'\n"
                "- Input: 'Kubernetes networking tutorial' → Output: 'Kubernetes networking tutorial official documentation'\n\n"
                "User Query: {query}\n\n"
                "Output ONLY valid JSON with double quotes, no markdown, no code blocks:\n"
                "{\"topic\": \"main entity or technology\", \"enriched_query\": \"optimized search query\"}"
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
            enriched = data.get('enriched_query', query)
            topic = data.get('topic', '').lower()
            
            # Validation: Ensure topic name is present in enriched query
            # If topic is extracted but not in enriched query, prepend it
            if topic and topic not in enriched.lower():
                print(f"DEBUG: Topic '{topic}' not found in enriched query, prepending it")
                enriched = f"{topic} {enriched}"
            
            print(f"DEBUG: Extracted - Topic: {data.get('topic', 'unknown')}, Enriched: {enriched}")
            return enriched
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
                for r in results:
                    try:
                        all_results.append({
                            "title": r.get("title", "No title"),
                            "url": r.get("url", ""),
                            "content": r.get("content", "")
                        })
                    except (KeyError, TypeError) as e:
                        print(f"DEBUG: Error processing result: {e}")
                        continue
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
        # Use QueryExtractorAgent to extract and enrich the query
        enriched_query = query_extractor.extract_and_enrich(user_query)

        # Step 1: Broad search (now with enriched query)
        search_q = build_search_query(enriched_query)  # Pass enriched
        search_results = search_searxng(search_q)

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