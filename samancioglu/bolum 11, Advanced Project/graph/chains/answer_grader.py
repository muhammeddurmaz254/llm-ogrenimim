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

class GradeAnswer(BaseModel):
    """
    Binary score for hallucination present in generated answer.
    """

    binary_score : bool = Field(
        description="Answer adresses the question. True if it does, False otherwise."
        )

structured_llm_grader = llm.with_structured_output(GradeAnswer)

system_prompt = """
You are a grader assessing whether an answer addresses / resolves a question 
Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question.
"""

human_prompt = """
User question: \n\n {question} \n\n LLM generation: {generation}
"""

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", human_prompt)
    ]
)

answer_grader = answer_prompt | structured_llm_grader