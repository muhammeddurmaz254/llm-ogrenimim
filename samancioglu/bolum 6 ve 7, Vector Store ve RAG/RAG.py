from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import langchainhub as hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
import os
import bs4

load_dotenv()

llm = ChatOpenAI(
    model="gemini-3.6-flash",
    api_key = os.getenv("GEMINI_API_KEY"),
    base_url = os.getenv("GEMINI_BASE_URL")
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only= bs4.SoupStrainer(
            class_ = ("post-content", "post-title", "post-header")
        )
    )
)

docs = loader.load()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
splits = text_splitter.split_documents(docs)

vector_store = Chroma.from_documents(
    documents=splits,
    embedding=embeddings
)

retriever = vector_store.as_retriever()

#rlm/rag_prompt 
prompt = PromptTemplate.from_template("""
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
""")

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


if __name__ == "__main__":
    for chunk in rag_chain.stream("what is maximum inner product search?"):
        print(chunk, end="", flush=True)