from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langsmith import Client
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model="gemini-3.6-flash",
    api_key = os.getenv("GEMINI_API_KEY"),
    base_url = os.getenv("GEMINI_BASE_URL"),
    temperature=0
)

client = Client()
prompt = client.pull_prompt("rlm/rag-prompt")

generation_chain = prompt | llm | StrOutputParser()