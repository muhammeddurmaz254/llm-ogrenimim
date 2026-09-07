from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv
import os


load_dotenv()

llm = ChatOpenAI(
    model="gemini-3.6-flash",
    api_key = os.getenv("GEMINI_API_KEY"),
    base_url = os.getenv("GEMINI_BASE_URL"),
    temperature=0
)

class GradeDocuments(BaseModel):
    """
    Binary score for relevance check on retrieved documents.
    """

    binary_score : bool = Field(
        description="Documents are relevant to the user query or not. True if relevant, False otherwise."
        )

structured_llm_grader = llm.with_structured_output(GradeDocuments)

system_prompt = """
You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.
If the document contains keyword or semantic meaning related to question, grade it as relevant. 
Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts.
"""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "Retrieved Documents: {document} User Question: {question}")
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader

