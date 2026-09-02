from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os

load_dotenv()

llm = ChatOpenAI(
    model="gemini-3.6-flash",
    api_key = os.getenv("GEMINI_API_KEY"),
    base_url = os.getenv("GEMINI_BASE_URL")
)

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a helpful assistant. Answer all questions as best as you can."),
    MessagesPlaceholder(variable_name="messages"),
])

chain = prompt | llm
config = {"configurable": {"session_id": "session1"} }
with_message_history = RunnableWithMessageHistory(chain, get_session_history)


if __name__ == "__main__":
    while True:
        user_input = input(">")

        response = with_message_history.invoke(
            [
                HumanMessage(content=user_input)
            ],
            config=config
        )

        print(response.content)