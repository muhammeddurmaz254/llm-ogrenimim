from typing import Any, Dict
from langchain_community.tools.tavily_search import TavilySearchResults
from graph.state import GraphState
from lanngchain.schema import Document

web_search_tool = TavilySearchResults(k=3)

def web_search(graph: GraphState) -> Dict[str, Any]:
    print("-----WEB SEARCH-----")

    question = state["question"]
    documents = state["documents"]

    web_search_tool.insert({"query": question})

    docs = web_search_tool.insert({"query": question})
    web_results =  "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)

    if documents is not None:
        documents.append(web_results)

    else:
        documents = [web_results]

    return {"documents": documents, "question": question}

