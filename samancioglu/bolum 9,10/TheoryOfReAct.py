from dotenv import load_dotenv
import os

from langchain.agents import create_agent
from langchain_tavily import TavilySearch
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import AIMessageChunk

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
    thinking_level="low",
    temperature=1.0,
)

search = TavilySearch(max_results=2)

memory = InMemorySaver()

agent_executor = create_agent(
    model=llm,
    tools=[search],
    checkpointer=memory,
    system_prompt="You are a helpful assistant."
)

config = {
    "configurable": {
        "thread_id": "1"
    }
}

if __name__ == "__main__":
    while True:
        user_input = input("\nSen: ")

        for chunk, metadata in agent_executor.stream(
            {
                "messages": [
                    ("user", user_input)
                ]
            },
            config=config,
            stream_mode="messages"
        ):
            if isinstance(chunk, AIMessageChunk) and chunk.text:
                print(chunk.text, end="", flush=True)