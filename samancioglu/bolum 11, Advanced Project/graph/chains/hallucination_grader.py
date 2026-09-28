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

class GradeHallucination(BaseModel):
    """
    Binary score for hallucination present in generated answer.
    """

    binary_score : bool = Field(
        description="Answer is grounded in the facts. True if grounded, False otherwise."
        )

llm.with_structured_output(GradeHallucination)

structured_llm_grader = llm.with_structured_output(GradeHallucination)

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