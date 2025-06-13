import urllib.request
import urllib.parse
import feedparser
import os
import time
import re

SEARCH_TERMS = [
    "language learning",
    "glitch tokens", 
    "quantum biology",
    "quantum radar",
]
OUTPUT_DIR = "./arxiv_pdfs"
MAX_RESULTS = 10

def sanitize_filename(filename):
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    filename = re.sub(r'\s+', ' ', filename)
    return filename.strip()[:100]

def search_arxiv(query, max_results=50):
    try:
        print(f"Searching: {query}")
        
        base_url = 'https://export.arxiv.org/api/query?'
        params = {
            'search_query': f'all:{query}',
            'start': 0,
            'max_results': max_results,
            'sortBy': 'relevance',
            'sortOrder': 'descending'
        }
        
        query_string = '&'.join([f'{k}={urllib.parse.quote(str(v))}' for k, v in params.items()])
        url = base_url + query_string
        
        response = urllib.request.urlopen(url)
        feed = feedparser.parse(response.read().decode('utf-8'))
        
        papers = []
        for entry in feed.entries:
            paper_id = entry.id.split('/abs/')[-1]
            papers.append({
                'id': paper_id,
                'title': entry.title.replace('\n', ' ').strip(),
                'authors': [author.name for author in entry.authors],
                'published': entry.published,
                'pdf_url': f'https://arxiv.org/pdf/{paper_id}.pdf'
            })
        
        print(f"Found {len(papers)} papers")
        return papers
        
    except Exception as e:
        print(f"Search error: {e}")
        return []

def download_pdf(paper, output_dir):
    try:
        safe_title = sanitize_filename(paper['title'])
        filename = f"{paper['id']}_{safe_title}.pdf"
        filepath = os.path.join(output_dir, filename)
        
        if os.path.exists(filepath):
            print(f"Skip: {filename}")
            return True
        
        print(f"Download: {filename}")
        urllib.request.urlretrieve(paper['pdf_url'], filepath)
        print(f"✓ {filename}")
        return True
        
    except Exception as e:
        print(f"✗ {paper['id']}: {e}")
        return False

def download_arxiv_papers(search_term, output_dir, max_results=50):
    print(f"Term: {search_term} | Dir: {output_dir} | Max: {max_results}")
    
    os.makedirs(output_dir, exist_ok=True)
    papers = search_arxiv(search_term, max_results)
    
    if not papers:
        print("No papers found")
        return
    
    successful = 0
    for i, paper in enumerate(papers, 1):
        print(f"[{i}/{len(papers)}]", end=" ")
        if download_pdf(paper, output_dir):
            successful += 1
        time.sleep(1)
    
    print(f"\nDone: {successful}/{len(papers)} downloaded to {output_dir}")

if __name__ == "__main__":
    for term in SEARCH_TERMS:
        download_arxiv_papers(term, OUTPUT_DIR, MAX_RESULTS)