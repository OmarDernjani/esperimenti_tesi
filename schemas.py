from pydantic import BaseModel, Field
from datetime import datetime

class ProblemSchema(BaseModel):
    dataset: str
    question: str
    model_optimizer: str
    code : str
    difficulty: str
    input_output: str
    critique: str | None = None

class SolutionSchema(BaseModel):
    problem_id: int
    algorithm: str
    prompt : str
    model_target: str
    code: str
    accuracy: float
    timestamp: datetime = Field(default_factory = datetime.now)
