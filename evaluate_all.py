import os
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from sqlalchemy import select
from tqdm import tqdm
from db import engine
from models import Problem, Solution
from eval import create_script_and_test

load_dotenv()

def evaluate_database():
    with Session(engine) as session:
        stmt = select(Problem.problem_id)
        result = session.execute(stmt).scalars().all()
        
        if not result:
            print("Nessun problema trovato nel database.")
            return

        print(f"Trovati {len(result)} problemi. Inizio la valutazione...\n")
        for problem_id in tqdm(result, desc="Valutazione in corso"):
            create_script_and_test(problem_id)

if __name__ == "__main__":
    evaluate_database()
    print("\n Valutazione completata su tutto il database")