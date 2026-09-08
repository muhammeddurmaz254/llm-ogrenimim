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

class GradeHallucination(BaseModel):
    """
    Binary score for hallucination present in generated answer.
    """

    binary_score : bool = Field(
        description="Answer is grounded in the facts. True if grounded, False otherwise."
        )

llm.with_structured_output(GradeHallucination)

structured_llm_grader = llm.with_name(GradeHallucination)

system_prompt = """
You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.
Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts.
"""

human_prompt = """
Set of facts: \n\n {documents} \n\n LLM generation: {generation}
"""

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", human_prompt)
    ]
)

hallucination_grader = hallucination_prompt | structured_llm_grader