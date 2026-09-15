from node_constants import GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEB_SEARCH
from graph.nodes import generate, grade_documents, web_search, retrieve
from graph.chains.router import question_router, RouteQuery
from graph.state import GraphState
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.answer_grader import answer_grader
from langgraph.graph import END, StateGraph
from dotenv import load_dotenv

load_dotenv()

workflow = StateGraph(GraphState)

workflow.add_node(GENERATE, generate)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(WEB_SEARCH, web_search)


app = workflow.compile()
app.get_graph().draw_mermaid_png(output_file="graph.png")