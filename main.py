import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# # Uncomment if you want to temporarily disable logger
# logger = logging.getLogger()
# logger.disabled = True

import nest_asyncio

nest_asyncio.apply()

from llama_index import (
    SimpleDirectoryReader,
    ServiceContext,
    get_response_synthesizer,
)

from llama_index.indices.document_summary import DocumentSummaryIndex
from llama_index.llms import OpenAI

wiki_titles = ["Toronto", "Seattle", "Chicago", "Boston", "Houston"]

from pathlib import Path

import requests 

for title in wiki_titles:
    response = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action":"query",
            "format": "json",
            "titles": title,
            "prop": "extracts",
            # 'exintro': True,
            "explaintext": True,
        },
    ).json()
    page = next(iter(response["query"]["pages"].values()))
    wiki_text = page["extract"]

    data_path = Path("data")
    if not data_path.exists():
        Path.mkdir(data_path)

    with open(data_path / f"{title}.txt", "w") as fp: 
        fp.write(wiki_text)

# Load all wiki documents
# Each txt file is its own Document object?
city_docs = []
for wiki_title in wiki_titles:
    docs = SimpleDirectoryReader(
        input_files=[f"data/{wiki_title}.txt"]
    ).load_data()
    docs[0].doc_id = wiki_title
    city_docs.extend(docs)
print(len(city_docs))