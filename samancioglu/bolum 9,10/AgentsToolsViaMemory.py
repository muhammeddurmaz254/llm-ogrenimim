from dotenv import load_dotenv
import os

from langchain.agents import create_agent
from langchain_tavily import TavilySearch
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver

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
    system_prompt=(
        "Sen yardımcı bir asistansın. Güncel bilgi isteyen sorularda "
        "Tavily arama aracını kullan."
    ),
)

config = {
    "configurable": {
        "thread_id": "1"
    }
}


if __name__ == "__main__":
    while True:
        user_input = input("\nSen: ").strip()

        if user_input.lower() in {"exit", "quit", "çıkış"}:
            print("Program kapatıldı.")
            break

        if not user_input:
            continue

        try:
            result = agent_executor.invoke(
                {
                    "messages": [
                        ("user", user_input)
                    ]
                },
                config=config,
            )

            # Sadece mevcut konuşma turunun mesajlarını al
            current_turn_messages = []

            for message in reversed(result["messages"]):
                current_turn_messages.append(message)

                if getattr(message, "type", None) == "human":
                    break

            current_turn_messages.reverse()

            tool_used = False

            for message in current_turn_messages:
                tool_calls = getattr(message, "tool_calls", None)

                if tool_calls:
                    tool_used = True

                    for tool_call in tool_calls:
                        print("\n[TOOL ÇAĞRILDI]")
                        print(f"Adı: {tool_call['name']}")
                        print(f"Parametreler: {tool_call['args']}")

                if getattr(message, "type", None) == "tool":
                    print("\n[TOOL SONUCU]")
                    print(f"Tool: {getattr(message, 'name', 'Bilinmiyor')}")
                    print(f"Sonuç: {message.content}")

            print(
                f"\nTool kullanıldı mı? "
                f"{'EVET' if tool_used else 'HAYIR'}"
            )

            print(f"\nAsistan: {result['messages'][-1].content}")

        except Exception as error:
            print(f"\nBir hata oluştu: {type(error).__name__}")
            print(error)