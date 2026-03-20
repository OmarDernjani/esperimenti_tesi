import py_compile
import os
import json
import ast
import tempfile
from db import engine
from models import Problem, Solution
from sqlalchemy.orm import Session
from utils import run_and_test_code, calculate_pass_at_k


def get_function_names(code: str) -> list[str]:
    tree = ast.parse(code)
    return [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

def check_syntax(path: str) -> bool:
    """Checks for syntax errors without executing the code."""
    try:
        py_compile.compile(path, doraise=True)
        return True
    except py_compile.PyCompileError as e:
        print(f"Syntax error in {path}: {e}")
        return False


def evaluate_script(file_path: str, io_data: dict) -> dict:
    """
    Scorre tutti gli input, li testa contro lo script e confronta 
    i risultati con gli output attesi.
    """
    inputs = io_data.get("inputs", [])
    outputs = io_data.get("outputs", [])
    
    if len(inputs) != len(outputs):
        return {"error": "Il numero di input non corrisponde al numero di output."}
    
    results = []
    passed_all = True
    
    for i, (test_in, expected_out) in enumerate(zip(inputs, outputs)):
        esito = run_and_test_code(file_path, test_in)
        
        if esito["success"]:
            actual_out_clean = "\n".join(line.rstrip() for line in esito["output"].splitlines()).strip()
            expected_out_clean = "\n".join(line.rstrip() for line in expected_out.splitlines()).strip()

            if actual_out_clean == expected_out_clean:
                results.append({"test_case": i, "passed": True})
            else:
                passed_all = False
                results.append({
                    "test_case": i,
                    "passed": False,
                    "expected": expected_out_clean,
                    "actual": actual_out_clean
                })
        else:
            passed_all = False
            results.append({
                "test_case": i,
                "passed": False,
                "error": esito["error"]
            })
            
    return {"passed_all": passed_all, "details": results}



def create_script_and_test(problem_id: int):
    with Session(engine) as session:

        problem = session.get(Problem, problem_id)
        if problem is None:
            print(f"Problema {problem_id} non trovato.")
            return

        solutions = list(problem.solutions)  # eager load per garantire il tracking

        if not solutions:
            print(f"Problema {problem_id}: nessuna soluzione, skip.")
            return

        try:
            io_dict = json.loads(problem.input_output)
            total_tests = len(io_dict.get("inputs", []))
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Problema {problem_id}: input_output non valido ({e}), skip.")
            return

        if total_tests == 0:
            print(f"Problema {problem_id}: nessun test case, skip.")
            return

        n_total = len(solutions)
        c_correct = 0

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                for sol in solutions:
                    sol_path = os.path.join(temp_dir, f"sol_{sol.sol_id}.py")

                    with open(sol_path, "w", encoding="utf-8") as f:
                        f.write(sol.code)

                    report = evaluate_script(sol_path, io_dict)

                    passed_tests = sum(1 for d in report.get("details", []) if d["passed"])
                    individual_accuracy = passed_tests / total_tests

                    sol.accuracy = individual_accuracy

                    if report.get("passed_all"):
                        c_correct += 1
                        print(f"  Sol {sol.sol_id}: Passata (Accuracy: {individual_accuracy:.2f})")
                    else:
                        print(f"  Sol {sol.sol_id}: Fallita  (Accuracy: {individual_accuracy:.2f})")

        except Exception as e:
            print(f"Problema {problem_id}: errore durante valutazione ({e}), rollback.")
            session.rollback()
            return

        session.commit()
        print(f"Problema {problem_id}: accuratezze salvate.")

        for k in [1, 3, 5, 10]:
            if n_total >= k:
                pass_k_score = calculate_pass_at_k(n_total, c_correct, k)
                print(f"  pass@{k}: {pass_k_score:.4f}")