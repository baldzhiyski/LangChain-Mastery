from ragatouille import RAGPretrainedModel
import requests

# Load a pretrained ColBERTv2.0 model for retrieval-augmented generation.
# This model is optimized for dense passage retrieval.
RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

def get_wikipedia_page(title: str):
    """
    Retrieve the full text content of a Wikipedia page.

    :param title: str - Title of the Wikipedia page.
    :return: str - Full text content of the page as raw string.
    """
    # Wikipedia API endpoint
    URL = "https://en.wikipedia.org/w/api.php"

    # Parameters for the API request:
    # - action=query: fetch page data
    # - format=json: return JSON
    # - titles=title: specify which page
    # - prop=extracts: return plain-text content (not HTML)
    # - explaintext=True: ensures we only get text, no markup
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
    }

    # Custom User-Agent header (good practice so Wikipedia knows who is calling)
    headers = {"User-Agent": "RAGatouille_tutorial/0.0.1 (ben@clavie.eu)"}

    # Make the GET request
    response = requests.get(URL, params=params, headers=headers)
    data = response.json()

    # Extract the actual text of the page
    # "pages" is a dict with page IDs as keys, so we grab the first one
    page = next(iter(data["query"]["pages"].values()))
    return page["extract"] if "extract" in page else None


# Fetch the raw text of the "Hayao Miyazaki" Wikipedia page
full_document = get_wikipedia_page("Hayao_Miyazaki")

# Index the retrieved document using the RAG model
RAG.index(
    collection=[full_document],    # A list of documents to index (here just 1 Wikipedia page)
    index_name="Miyazaki-123",     # Name of the index (used to reference later)
    max_document_length=180,       # Split long documents into chunks of ~180 tokens
    split_documents=True,          # Enable automatic splitting of documents
)

results = RAG.search(query="What animation studio did Miyazaki found?", k=3)

# Inspect and print the top-k results
for i, r in enumerate(results, 1):
    # r may contain fields like 'text'/'document'/'passage' depending on version.
    # Try common keys defensively:
    text = r.get("text") or r.get("passage") or r.get("document") or str(r)
    print(f"Result {i}:\n{text[:500]}...\n")

