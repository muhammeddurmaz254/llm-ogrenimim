from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os



load_dotenv()

llm = ChatOpenAI(
    model="gemini-3.6-flash",
    api_key = os.getenv("GEMINI_API_KEY"),
    base_url = os.getenv("BASE_URL")
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

if __name__ == "__main__":
    print(chain.invoke({
        "language": "Turkish",
        "text": "Hello, world!"
    }))