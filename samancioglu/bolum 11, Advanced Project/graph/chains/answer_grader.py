from langchain_openai import ChatOpenAI
from graph.chains.rate_limiter import rate_limiter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv
import os


load_dotenv()

llm = ChatOpenAI(
    model="nvidia/nemotron-3-super-120b-a12b",
    api_key = os.getenv("NVIDIA_API_KEY"),
    base_url = "https://integrate.api.nvidia.com/v1",
    temperature=0,
    rate_limiter=rate_limiter
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