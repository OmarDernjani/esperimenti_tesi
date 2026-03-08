from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import DeclarativeBase, relationship
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

class Base(DeclarativeBase):
    pass

class Problem(Base):
    __tablename__ = "problems"
    problem_id = Column(Integer, primary_key=True, autoincrement=True)
    dataset = Column(String, nullable=False)
    question = Column(Text, nullable=False)
    model_optimizer = Column(String, nullable = False)
    code = Column(Text, nullable = False)
    difficulty = Column(String, nullable=False)
    input_output = Column(Text, nullable=False)
    solutions = relationship("Solution", back_populates="problem")

class Solution(Base):
    __tablename__ = "solutions"
    sol_id = Column(Integer, primary_key=True, autoincrement=True)
    problem_id = Column(Integer, ForeignKey("problems.problem_id"), nullable=False, unique = False)
    algorithm = Column(String, nullable=False)
    prompt = Column(Text, nullable = False)
    model_target = Column(String, nullable = False)
    code = Column(Text, nullable = False)
    accuracy = Column(Float, nullable = False)
    timestamp = Column(DateTime, default = datetime.now)
    problem = relationship("Problem", back_populates="solutions")

engine = create_engine(os.getenv("DATABASE_URL"))
Base.metadata.create_all(engine)
