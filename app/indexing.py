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
import re  # ADD THIS IMPORT - This was missing!

# Load environment variables from .env file
dotenv.load_dotenv()
persist_directory = "./chroma_langchain_db"

# Source URLs
source_urls = {
    # Pages that contain multiple article links
    "index_pages": [
        "https://www.cdc.gov/media/releases/index.html",
        "https://www.nih.gov/news-events/news-releases",
        "https://www.who.int/news",
    ],
    
    # Pages that are direct articles
    "direct_articles": [
        # Remove the example URL that returns 404
        # "https://medlineplus.gov/news/fullstory_123456.html",
    ]
}


# -------- Prefect Tasks -------- #

@task(name="discover_new_documents")
def discover_new_documents(index_urls: list[str]) -> list[str]:
    """Scrape index pages to collect article links, filtering only real news/article URLs."""
    article_links = []
    
    for index_url in index_urls:
        try:
            res = requests.get(index_url, timeout=10)
            res.raise_for_status()
            soup = BeautifulSoup(res.text, "html.parser")

            for a in soup.find_all("a", href=True):
                href = a["href"]

                # Normalize to absolute URL
                if not href.startswith("http"):
                    from urllib.parse import urljoin
                    href = urljoin(index_url, href)

                # ---- Filtering rules ---- #
                if "cdc.gov" in href and re.search(r"/media/releases/", href):
                    article_links.append(href)
                elif "nih.gov" in href and re.search(r"/news-events/news-releases/", href):
                    article_links.append(href)
                elif "who.int" in href and re.search(r"/news/item/", href):
                    article_links.append(href)

        except Exception as e:
            print(f"Error scraping {index_url}: {e}")
    
    return list(set(article_links))  # deduplicate


@task(name="check_existing_documents")
def check_existing_documents(urls: list[str]) -> tuple[list[str], list[str]]:
    """
    Check which URLs already exist in the vector store.
    Returns: (new_urls, existing_urls)
    """
    try:
        # Initialize vector store to check existing documents
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        vector_store = Chroma(
            collection_name="example_collection",
            embedding_function=embeddings,
            persist_directory=persist_directory
        )
        
        # Get all existing document sources
        collection = vector_store._collection
        existing_docs = collection.get()
        existing_sources = set()
        
        if existing_docs and existing_docs.get('metadatas'):
            for metadata in existing_docs['metadatas']:
                if metadata and 'source' in metadata:
                    existing_sources.add(metadata['source'])
        
        new_urls = []
        existing_urls = []
        
        for url in urls:
            if url in existing_sources:
                existing_urls.append(url)
            else:
                new_urls.append(url)
        
        print(f"Found {len(existing_urls)} existing URLs, {len(new_urls)} new URLs")
        return new_urls, existing_urls
        
    except Exception as e:
        print(f"Error checking existing documents: {e}")
        # If we can't check, process all URLs to be safe
        return urls, []


@task(name="validate_document_urls")
def validate_document_urls(urls: list[str]) -> list[str]:
    """
    Validate the URLs of documents to ensure they are accessible and relevant.
    - Removes URLs that return non-200 responses
    - Skips URLs with invalid content types (e.g., images)
    """
    validated_urls = []
    for url in urls:
        try:
            r = httpx.get(url, timeout=10)
            if r.status_code == 200 and "text/html" in r.headers.get("content-type", ""):
                validated_urls.append(url)
            else:
                print(f"Skipping {url} -> status {r.status_code}, content {r.headers.get('content-type')}")
        except Exception as e:
            print(f"Error validating {url}: {e}")
    return validated_urls


@task(name="process_documents")
def process_documents(urls: list[str]) -> list[Document]:
    """
    Fetch the HTML content from URLs, extract readable text, 
    and convert into LangChain Document objects.
    """
    documents = []

    for url in urls:
        try:
            r = httpx.get(url, timeout=15)
            if r.status_code != 200:
                print(f"Skipping {url} -> {r.status_code}")
                continue

            soup = BeautifulSoup(r.text, "html.parser")

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Basic text extraction (you can refine per-site)
            text = soup.get_text(separator=" ", strip=True)

            if len(text) < 500:  # skip junk pages
                print(f"Skipping {url} -> too little text ({len(text)} chars)")
                continue

            # Clean up the text
            text = re.sub(r'\s+', ' ', text).strip()

            # Wrap into LangChain Document
            doc = Document(
                page_content=text,
                metadata={"source": url, "length": len(text)}
            )
            documents.append(doc)
            print(f"Processed {url} -> {len(text)} characters")

        except Exception as e:
            print(f"Error processing {url}: {e}")

    print(f"Total documents processed: {len(documents)}")
    return documents


@task(name="add_documents_to_vector_store")
def add_documents(documents: list[Document]):
    """Write processed documents into vector store"""
    if not documents:
        print("No documents to add to vector store")
        return
    
    print(f"Adding {len(documents)} documents to vector store")
    add_documents_to_vector_store(documents=documents)


# -------- Prefect Flow -------- #

@flow(name="index")
def index_documents():
    # Call tasks
    article_links = discover_new_documents(source_urls["index_pages"])
    direct_articles = source_urls["direct_articles"]

    # Prefect will pass task result as a value here
    all_urls = article_links + direct_articles  
    print(f"Found {len(all_urls)} total URLs")

    if not all_urls:
        print("No URLs found to process")
        return

    # Check for existing documents first
    new_urls, existing_urls = check_existing_documents(all_urls)
    
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
    index_documents()
    print("Indexing completed.")