from langchain_openai import ChatOpenAI
from graph.chains.rate_limiter import rate_limiter
from langchain_core.output_parsers import StrOutputParser
from langsmith import Client
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model="nvidia/nemotron-3-super-120b-a12b",
    api_key = os.getenv("NVIDIA_API_KEY"),
    base_url = "https://integrate.api.nvidia.com/v1",
    temperature=0,
    rate_limiter=rate_limiter
)

client = Client()
prompt = client.pull_prompt("rlm/rag-prompt")

generation_chain = prompt | llm | StrOutputParser()