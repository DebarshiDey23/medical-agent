#!/usr/bin/env python3
"""
Tool to dynamically add new sources to the indexing pipeline
"""

import yaml
import argparse
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import re
from pathlib import Path


def analyze_site_structure(url: str) -> dict:
    """
    Analyze a website to suggest link patterns and discover news sections
    """
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find potential news/press sections
        news_links = []
        patterns = set()
        
        for a in soup.find_all('a', href=True):
            href = a.get('href')
            text = a.get_text().lower().strip()
            
            # Look for news-related links
            if any(keyword in text for keyword in ['news', 'press', 'media', 'release', 'blog', 'updates']):
                if not href.startswith('http'):
                    href = urljoin(url, href)
                news_links.append({'url': href, 'text': text})
                
                # Extract potential patterns
                path = urlparse(href).path
                # Look for date patterns
                date_pattern = re.search(r'/(\d{4})/(\d{1,2})/(\d{1,2})/', path)
                if date_pattern:
                    patterns.add('/\\d{4}/\\d{1,2}/\\d{1,2}/')
                
                # Look for year patterns
                year_pattern = re.search(r'/(\d{4})/', path)
                if year_pattern:
                    patterns.add('/\\d{4}/')
        
        return {
            'news_sections': news_links[:10],  # Top 10
            'suggested_patterns': list(patterns),
            'domain': urlparse(url).netloc
        }
        
    except Exception as e:
        print(f"Error analyzing {url}: {e}")
        return {}


def add_source_to_config(config_path: str, category: str, name: str, 
                        index_pages: list, patterns: list = None):
    """
    Add a new source to the configuration file
    """
    config_file = Path(config_path)
    
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {'sources': {}}
    
    if 'sources' not in config:
        config['sources'] = {}
    
    if category not in config['sources']:
        config['sources'][category] = {}
    
    # Auto-detect domain and patterns if not provided
    if index_pages:
        domain = urlparse(index_pages[0]).netloc
        
        if not patterns:
            print(f"Analyzing {index_pages[0]} for patterns...")
            analysis = analyze_site_structure(index_pages[0])
            patterns = analysis.get('suggested_patterns', [])
            print(f"Suggested patterns: {patterns}")
    
    config['sources'][category][name] = {
        'index_pages': index_pages,
        'link_patterns': patterns or ['/news/', '/press/'],
        'domain': domain
    }
    
    # Save updated config
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"Added source '{name}' to category '{category}'")
    return config


def discover_medical_sites(query: str, num_results: int = 10) -> list:
    """
    Use search to discover potential medical news sites
    """
    # This would integrate with search APIs to find medical news sites
    # For now, return some well-known medical sites
    known_medical_sites = [
        "https://www.webmd.com/news",
        "https://www.healthline.com/health-news",
        "https://www.medicalnewstoday.com",
        "https://www.statnews.com",
        "https://www.reuters.com/business/healthcare-pharmaceuticals/",
        "https://apnews.com/hub/health",
        "https://www.nature.com/nature/articles",
        "https://www.science.org/news",
        "https://www.newscientist.com/subject/health/",
        "https://www.scientificamerican.com/health/"
    ]
    
    print(f"Discovered {len(known_medical_sites)} potential medical news sources")
    return known_medical_sites


def interactive_source_addition():
    """
    Interactive CLI for adding sources
    """
    config_path = "sources.yaml"
    
    print("=== Medical News Source Addition Tool ===")
    print()
    
    while True:
        print("Options:")
        print("1. Add a single source")
        print("2. Analyze a website for patterns")
        print("3. Discover medical news sites")
        print("4. Exit")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == "1":
            category = input("Category (e.g., medical_news, research_institutions): ")
            name = input("Source name (e.g., cnn_health): ")
            urls_input = input("Index page URLs (comma-separated): ")
            urls = [url.strip() for url in urls_input.split(",") if url.strip()]
            
            patterns_input = input("Link patterns (comma-separated, optional): ")
            patterns = [p.strip() for p in patterns_input.split(",") if p.strip()] if patterns_input else None
            
            add_source_to_config(config_path, category, name, urls, patterns)
            
        elif choice == "2":
            url = input("Website URL to analyze: ").strip()
            if url:
                analysis = analyze_site_structure(url)
                print(f"\nAnalysis for {url}:")
                print(f"Domain: {analysis.get('domain')}")
                print(f"Suggested patterns: {analysis.get('suggested_patterns')}")
                print("News sections found:")
                for section in analysis.get('news_sections', []):
                    print(f"  - {section['text']}: {section['url']}")
        
        elif choice == "3":
            query = input("Search query (optional): ") or "medical news"
            sites = discover_medical_sites(query)
            print(f"\nDiscovered sites:")
            for i, site in enumerate(sites, 1):
                print(f"{i:2d}. {site}")
            
            selection = input("\nEnter numbers to add (comma-separated): ")
            if selection:
                try:
                    indices = [int(i.strip()) - 1 for i in selection.split(",")]
                    selected_sites = [sites[i] for i in indices if 0 <= i < len(sites)]
                    
                    if selected_sites:
                        category = input("Category for these sites: ") or "discovered_sites"
                        for site in selected_sites:
                            domain = urlparse(site).netloc.replace('www.', '').replace('.', '_')
                            add_source_to_config(config_path, category, domain, [site])
                            
                except (ValueError, IndexError):
                    print("Invalid selection")
        
        elif choice == "4":
            break
        
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add sources to the indexing pipeline")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--category", "-c", help="Source category")
    parser.add_argument("--name", "-n", help="Source name")
    parser.add_argument("--url", "-u", help="Source URL")
    parser.add_argument("--analyze", "-a", help="Analyze URL for patterns")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_source_addition()
    elif args.analyze:
        analysis = analyze_site_structure(args.analyze)
        print(yaml.dump(analysis, default_flow_style=False))
    elif args.category and args.name and args.url:
        add_source_to_config("sources.yaml", args.category, args.name, [args.url])
    else:
        print("Use --interactive for interactive mode, or provide --category, --name, and --url")