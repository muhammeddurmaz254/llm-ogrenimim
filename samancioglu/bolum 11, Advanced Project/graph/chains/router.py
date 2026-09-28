from langchain_openai import ChatOpenAI
from graph.chains.rate_limiter import rate_limiter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv
import os

load_dotenv()

class RouteQuery(BaseModel):
    """
    Route a user query to the most relevant datasoruce.
    """

    datasource: Literal["vectorstore", "websearch"] = Field(
        ..., 
        description="Given a user question choose to route it to web search or a vectorestore"
    )

llm = ChatOpenAI(
    model="nvidia/nemotron-3-super-120b-a12b",
    api_key = os.getenv("NVIDIA_API_KEY"),
    base_url = "https://integrate.api.nvidia.com/v1",
    temperature=0,
    rate_limiter=rate_limiter
)

structured_llm_router = llm.with_structured_output(RouteQuery)

system_prompt = """
You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to agents, prompt engineering and adversial attacks on llms.
Use the vectorstore for questions on these topics. For all else use web search.
"""

route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{question}")
    ]
)

question_router = route_prompt | structured_llm_router
