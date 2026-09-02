from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

llm = ChatOpenAI(
    model="gemini-3.6-flash",
    api_key = os.getenv("GEMINI_API_KEY"),
    base_url = os.getenv("GEMINI_BASE_URL")
)


messages = [
    HumanMessage(content="Hello, My name is Muhammed."),
    AIMessage(content="hello Muhammed, nice to meet you!"),
    HumanMessage(content="What is my name?")
]

parser = StrOutputParser()

chain = llm | parser

if __name__ == "__main__":
    print(chain.invoke(messages))