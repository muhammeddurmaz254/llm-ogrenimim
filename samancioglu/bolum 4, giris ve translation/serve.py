from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from fastapi import FastAPI
from langserve import add_routes


load_dotenv()

llm = ChatOpenAI(
    model="gemini-3.6-flash",
    api_key = os.getenv("GEMINI_API_KEY"),
    base_url = os.getenv("GEMINI_BASE_URL")
)


"""
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is the capital of France?")
]
"""


system_prompt = "Translate the following English text to {language}."
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{text}")
])

parser = StrOutputParser()

chain = prompt_template | llm | parser


app = FastAPI(
    title="Tasnlator APP",
    description="This is a simple translator app using FastAPI and LangChain.",
    version="1.0.0"
)

add_routes(
    app,
    chain,
    path = "/chain"
)



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)