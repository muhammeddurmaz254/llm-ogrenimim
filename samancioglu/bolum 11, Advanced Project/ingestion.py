from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
import bs4
import os

load_dotenv()

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/"
]

# calisma dizininden bagimsiz olarak proje klasorune kaydet
PERSIST_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".chroma")
COLLECTION_NAME = "rag-chroma-lilianweng"

embeddings = NVIDIAEmbeddings(
    model="nvidia/nemotron-3-embed-1b",
    api_key=os.getenv("NVIDIA_API_KEY"),
    truncate="END",
)

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embeddings,
)

# embedding sadece veritabani bossa yapilir, sonrasinda diskteki veritabani kullanilir
if not vector_store.get(limit=1)["ids"]:
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=2000,
        chunk_overlap=200
    )

    splits = text_splitter.split_documents(docs_list)

    try:
        vector_store.add_documents(splits)
    except Exception:
        # yarim kalan veritabani bir sonraki calistirmada embedding'i engellemesin
        vector_store.delete_collection()
        raise

retriever = vector_store.as_retriever()
