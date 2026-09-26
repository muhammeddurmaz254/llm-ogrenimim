from node_constants import GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEB_SEARCH
from graph.nodes import generate, grade_documents, web_search, retrieve
from graph.chains.router import question_router, RouteQuery
from graph.state import GraphState
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.answer_grader import answer_grader
from langgraph.graph import END, StateGraph
from dotenv import load_dotenv

load_dotenv()

def decide_to_generate(state: GraphState):
    print("-----ASSESS GRADED DOCUMENTS-----")
    if state["web_search"]:
        print(WEB_SEARCH)
        return WEB_SEARCH

    else: 
        return GENERATE


def grade_generations_grounded_in_documents_and_question(state: GraphState)-> str:
    print("-----Check Hallucinations-----")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {
            "documents": documents,
            "generation": generation
        }
    )

    if hallucination_grade := score.binary_score:
        print("GENERATION IS GROUNDED IN DOCUMENTS")
        score=answer_grader.invoke(
            {
                "question": question,
                "generation": generation
            }
            )

            if answer_grade := score.binary_score:
                print("GENERATION ADRESSES QUESTION")
                return "useful"
            else:
                print("GENERATION DOES NOT ADDRESS QUESTION")
                return "not useful"

    else:
            print("GENERATION IS NOT GROUNDED IN DOCUMENTS")
            return "not useful"

    
def route_question(state: GraphState) -> str:
    print("-----ROUTING QUESTION-----")
    question = state["question"]
    source = question_router.invoke({"question": question})
    if source.data_source == "websearch":
        print("WEBSEARCH")
        return WEBSEARCH
    elif source.data_source == "vectorstore":
        print("VECTORSTORE")
        return RETRIEVE

workflow = StateGraph(GraphState)

workflow.add_node(GENERATE, generate)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(WEB_SEARCH, web_search)


app = workflow.compile()
app.get_graph().draw_mermaid_png(output_file="graph.png")