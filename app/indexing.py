import chromadb
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import dotenv
from uuid import uuid4
from langchain_core.documents import Document
from prefect import flow, task
from vector_store import add_documents_to_vector_store
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
import httpx
import re
import yaml
import json
from pathlib import Path
from typing import Dict, List, Any
from urllib.parse import urljoin, urlparse
import os

# Load environment variables from .env file
dotenv.load_dotenv()
persist_directory = "./chroma_langchain_db"

# -------- Configuration Loading -------- #

def load_source_config(config_path: str = "sources.yaml") -> Dict[str, Any]:
    """Load source configuration from YAML file"""
    config_file = Path(config_path)
    
    if not config_file.exists():
        # Create default config if it doesn't exist
        default_config = {
            "sources": {
                "medical_news": {
                    "cdc": {
                        "index_pages": ["https://www.cdc.gov/media/releases/index.html"],
                        "link_patterns": [r"/media/releases/\d{4}/"],
                        "domain": "cdc.gov"
                    },
                    "nih": {
                        "index_pages": ["https://www.nih.gov/news-events/news-releases"],
                        "link_patterns": [r"/news-events/news-releases/\d{4}/"],
                        "domain": "nih.gov"
                    },
                    "who": {
                        "index_pages": ["https://www.who.int/news"],
                        "link_patterns": [r"/news/item/"],
                        "domain": "who.int"
                    }
                }
            },
            "discovery": {
                "auto_discover_similar_sites": True,
                "max_depth": 2,
                "respect_robots_txt": True,
                "user_agent": "Medical-Agent-Indexer/1.0"
            },
            "filtering": {
                "min_content_length": 500,
                "exclude_patterns": [
                    r"\.pdf$", r"\.doc$", r"\.xlsx?$",
                    r"/search\?", r"/tag/", r"/category/"
                ],
                "required_keywords": ["health", "medical", "disease", "treatment", "research"]
            }
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        
        print(f"Created default configuration at {config_path}")
        return default_config
    
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def load_source_config_from_env() -> Dict[str, Any]:
    """Load configuration from environment variables"""
    config = {
        "sources": {},
        "discovery": {
            "auto_discover_similar_sites": os.getenv("AUTO_DISCOVER", "true").lower() == "true",
            "max_depth": int(os.getenv("MAX_CRAWL_DEPTH", "2")),
            "respect_robots_txt": True,
            "user_agent": os.getenv("USER_AGENT", "Medical-Agent-Indexer/1.0")
        }
    }
    
    # Load URLs from environment variables
    source_urls = os.getenv("SOURCE_URLS", "")
    if source_urls:
        urls = [url.strip() for url in source_urls.split(",") if url.strip()]
        config["sources"]["custom"] = {
            "index_pages": urls,
            "link_patterns": [r".*"],  # Accept all patterns
            "domain": "*"
        }
    
    return config


# -------- Dynamic URL Discovery Tasks -------- #

@task(name="discover_related_sites")
def discover_related_sites(seed_urls: List[str], keywords: List[str]) -> List[str]:
    """
    Discover related medical/health sites by analyzing links from seed sites
    """
    related_sites = set()
    
    for seed_url in seed_urls:
        try:
            response = requests.get(seed_url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Look for external links
            for link in soup.find_all('a', href=True):
                href = link.get('href')
                if not href or not href.startswith('http'):
                    continue
                
                domain = urlparse(href).netloc
                
                # Check if link text or surrounding context contains health keywords
                link_text = link.get_text().lower()
                if any(keyword in link_text for keyword in keywords):
                    # Check if it's a news/press release section
                    if any(pattern in href.lower() for pattern in ['news', 'press', 'release', 'media']):
                        related_sites.add(href)
                        
        except Exception as e:
            print(f"Error discovering related sites from {seed_url}: {e}")
    
    return list(related_sites)


@task(name="discover_sitemap_urls")
def discover_sitemap_urls(base_domains: List[str]) -> List[str]:
    """
    Discover URLs from sitemaps
    """
    sitemap_urls = []
    
    for domain in base_domains:
        if not domain.startswith('http'):
            domain = f"https://{domain}"
        
        # Common sitemap locations
        sitemap_paths = ['/sitemap.xml', '/sitemap_index.xml', '/robots.txt']
        
        for path in sitemap_paths:
            try:
                sitemap_url = urljoin(domain, path)
                response = requests.get(sitemap_url, timeout=10)
                
                if response.status_code == 200:
                    if path == '/robots.txt':
                        # Parse robots.txt for sitemap references
                        for line in response.text.split('\n'):
                            if line.lower().startswith('sitemap:'):
                                sitemap_urls.append(line.split(':', 1)[1].strip())
                    else:
                        # Parse XML sitemap
                        soup = BeautifulSoup(response.text, 'xml')
                        for loc in soup.find_all('loc'):
                            url = loc.get_text()
                            # Filter for news/press release URLs
                            if any(keyword in url.lower() for keyword in ['news', 'press', 'release', 'media']):
                                sitemap_urls.append(url)
                                
            except Exception as e:
                print(f"Error processing sitemap for {domain}: {e}")
    
    return sitemap_urls


@task(name="discover_rss_feeds")
def discover_rss_feeds(domains: List[str]) -> List[str]:
    """
    Discover RSS/Atom feeds that might contain news articles
    """
    feed_urls = []
    
    for domain in domains:
        if not domain.startswith('http'):
            domain = f"https://{domain}"
        
        # Common RSS feed paths
        feed_paths = [
            '/rss', '/rss.xml', '/feed', '/feed.xml', '/feeds/all.atom.xml',
            '/news/rss', '/news/feed', '/press/rss', '/media/feed'
        ]
        
        try:
            # Also check HTML pages for feed links
            response = requests.get(domain, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Look for feed links in HTML head
            for link in soup.find_all('link', {'type': ['application/rss+xml', 'application/atom+xml']}):
                href = link.get('href')
                if href:
                    feed_urls.append(urljoin(domain, href))
            
            # Try common feed paths
            for path in feed_paths:
                feed_url = urljoin(domain, path)
                try:
                    feed_response = requests.head(feed_url, timeout=5)
                    if feed_response.status_code == 200:
                        feed_urls.append(feed_url)
                except:
                    pass
                    
        except Exception as e:
            print(f"Error discovering feeds for {domain}: {e}")
    
    return list(set(feed_urls))


@task(name="parse_rss_feeds")
def parse_rss_feeds(feed_urls: List[str]) -> List[str]:
    """
    Parse RSS feeds to extract article URLs
    """
    try:
        import feedparser
    except ImportError:
        print("feedparser not installed. Install with: pip install feedparser")
        return []
    
    article_urls = []
    
    for feed_url in feed_urls:
        try:
            feed = feedparser.parse(feed_url)
            
            for entry in feed.entries:
                if hasattr(entry, 'link') and entry.link:
                    article_urls.append(entry.link)
                    
        except Exception as e:
            print(f"Error parsing feed {feed_url}: {e}")
    
    return list(set(article_urls))


@task(name="smart_url_discovery")
def smart_url_discovery(config: Dict[str, Any]) -> List[str]:
    """
    Intelligently discover URLs using multiple strategies
    """
    all_urls = []
    
    # Strategy 1: Use configured sources
    for category, sources in config.get("sources", {}).items():
        for source_name, source_config in sources.items():
            if isinstance(source_config, dict):
                all_urls.extend(source_config.get("index_pages", []))
    
    # Strategy 2: Auto-discover similar sites if enabled
    if config.get("discovery", {}).get("auto_discover_similar_sites", False):
        keywords = config.get("filtering", {}).get("required_keywords", ["health", "medical"])
        seed_urls = all_urls[:3]  # Use first few as seeds
        related_sites = discover_related_sites(seed_urls, keywords)
        all_urls.extend(related_sites)
    
    # Strategy 3: Use sitemaps
    domains = []
    for url in all_urls:
        domain = urlparse(url).netloc
        if domain not in domains:
            domains.append(domain)
    
    sitemap_urls = discover_sitemap_urls(domains)
    all_urls.extend(sitemap_urls)
    
    # Strategy 4: Use RSS feeds
    rss_feeds = discover_rss_feeds(domains)
    rss_articles = parse_rss_feeds(rss_feeds)
    all_urls.extend(rss_articles)
    
    return list(set(all_urls))  # Deduplicate


@task(name="discover_new_documents_dynamic")
def discover_new_documents_dynamic(config: Dict[str, Any]) -> List[str]:
    """Enhanced document discovery using configuration"""
    article_links = []
    
    # Get URLs from smart discovery
    urls_to_crawl = smart_url_discovery(config)
    
    for url in urls_to_crawl:
        try:
            response = requests.get(url, timeout=10, 
                                  headers={'User-Agent': config.get('discovery', {}).get('user_agent', 'Mozilla/5.0')})
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract domain for pattern matching
            domain = urlparse(url).netloc

            for a in soup.find_all("a", href=True):
                href = a["href"]

                # Normalize to absolute URL
                if not href.startswith("http"):
                    href = urljoin(url, href)

                # Apply domain-specific patterns
                if should_include_url(href, config, domain):
                    article_links.append(href)

        except Exception as e:
            print(f"Error scraping {url}: {e}")
    
    return list(set(article_links))


def should_include_url(url: str, config: Dict[str, Any], domain: str) -> bool:
    """
    Determine if a URL should be included based on configuration
    """
    # Check exclude patterns
    exclude_patterns = config.get("filtering", {}).get("exclude_patterns", [])
    for pattern in exclude_patterns:
        if re.search(pattern, url, re.IGNORECASE):
            return False
    
    # Check domain-specific patterns
    for category, sources in config.get("sources", {}).items():
        for source_name, source_config in sources.items():
            if isinstance(source_config, dict):
                source_domain = source_config.get("domain", "")
                if source_domain == "*" or source_domain in domain:
                    patterns = source_config.get("link_patterns", [])
                    if any(re.search(pattern, url) for pattern in patterns):
                        return True
    
    # Default: include if it contains news-like keywords
    news_keywords = ['news', 'press', 'release', 'media', 'announcement']
    return any(keyword in url.lower() for keyword in news_keywords)


# -------- Main Flow -------- #

@flow(name="dynamic_index")
def index_documents_dynamic(config_source: str = "yaml"):
    """
    Main indexing flow with dynamic URL discovery
    """
    # Load configuration
    if config_source == "yaml":
        config = load_source_config("sources.yaml")
    elif config_source == "env":
        config = load_source_config_from_env()
    else:
        raise ValueError("config_source must be 'yaml' or 'env'")
    
    print(f"Loaded configuration: {len(config.get('sources', {}))} source categories")
    
    # Discover URLs dynamically
    article_links = discover_new_documents_dynamic(config)
    print(f"Discovered {len(article_links)} URLs")
    
    if not article_links:
        print("No URLs discovered")
        return

    # Check for existing documents
    from fixed_indexing import check_existing_documents, validate_document_urls, process_documents, add_documents
    
    new_urls, existing_urls = check_existing_documents(article_links)
    
    if existing_urls:
        print(f"Skipping {len(existing_urls)} existing documents")
    
    if not new_urls:
        print("No new URLs to process")
        return

    validated_urls = validate_document_urls(new_urls)
    print(f"Validated {len(validated_urls)} new URLs")
    
    if not validated_urls:
        print("No valid new URLs to process")
        return

    documents = process_documents(validated_urls)
    
    if documents:
        add_documents(documents)
        print(f"Successfully added {len(documents)} new documents to vector store")
    else:
        print("No new documents were processed successfully")


if __name__ == "__main__":
    # You can switch between configuration sources
    index_documents_dynamic(config_source="yaml")  # or "env"
    print("Dynamic indexing completed.")