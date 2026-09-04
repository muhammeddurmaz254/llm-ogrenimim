from dotenv import load_dotenv
import os

from langchain.agents import create_agent
from langchain_tavily import TavilySearch
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
    thinking_level="low",
    temperature=1.0,
)

search = TavilySearch(max_results=2)

agent_executor = create_agent(
    model=llm,
    tools=[search],
)

if __name__ == "__main__":
    response = agent_executor.invoke(
        {
            "messages": [
                (
                    "user",
                    "What is the current weather in Gebze, Kocaeli, Türkiye?"
                )
            ]
        }
    )

    print(response["messages"][-1].text)