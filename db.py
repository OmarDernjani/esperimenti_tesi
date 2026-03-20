from sqlalchemy import create_engine
import os
from models import Base

engine = create_engine(os.getenv("DATABASE_URL"))
Base.metadata.create_all(engine)