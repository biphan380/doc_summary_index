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
file_counter = 1
for wiki_title in wiki_titles:
    docs = SimpleDirectoryReader(
        input_files=[f"data/{wiki_title}.txt"]
    ).load_data()
    docs[0].doc_id = wiki_title
    city_docs.extend(docs)
    
    with open(f'doc_objects/city_docs_{file_counter}.txt', 'w') as f:
        for doc in city_docs:
            f.write(f'{doc.doc_id}\n')
            f.write(f'{doc.text}\n\n')
            f.write('======================== new doc object starts ================================\n')
    file_counter += 1

print(len(city_docs))

sys.exit()






# Build Document Summary Index
chatgpt = OpenAI(temperature=0, model="gpt-3.5-turbo")
service_context = ServiceContext.from_defaults(llm=chatgpt, chunk_size=1024)

# default mode of building the index
response_synthesizer = get_response_synthesizer(
    response_mode="tree_summarize", use_async=True
)
doc_summary_index = DocumentSummaryIndex.from_documents(
    city_docs,
    service_context=service_context,
    response_synthesizer=response_synthesizer,
    show_progress=True,
)

doc_summary_index.get_document_summary("Boston")

doc_summary_index.storage_context.persist("index")

from llama_index.indices.loading import load_index_from_storage
from llama_index import StorageContext

# rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir="index")
doc_summary_index = load_index_from_storage(storage_context)