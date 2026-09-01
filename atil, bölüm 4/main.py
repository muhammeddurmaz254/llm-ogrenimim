from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
import os
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    api_key = os.getenv("GEMINI_API_KEY"),
    base_url = os.getenv("BASE_URL"),
    temperature=0.7,
)

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is the capital of France?")
]

if __name__ == "__main__":
    response = llm.invoke(messages)
    print(response)