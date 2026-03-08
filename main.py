from dotenv import load_dotenv
import os
import utils
from tqdm import tqdm

load_dotenv()

train_data, _ = utils.load_apps_dataset()
problem = utils.get_problem(train_data, index=int(os.getenv("PROBLEM_INDEX", 0)))

solver_chain = utils.build_solver_chain(model=os.getenv("MODEL_TARGET"))
baseline_code = utils.extract_code(solver_chain.invoke({"user_prompt": problem["question"]}))

samples = utils.get_minibatch(
    train_data, 
    n_per_difficulty=int(os.getenv("N_PER_DIFFICULTY", 50))
)

for problem in tqdm(samples, desc="Problemi processati"):
    baseline_code = utils.extract_code(
        solver_chain.invoke({"user_prompt": problem["question"]})
    )

    variant_response = utils.resampling(
        question=problem["question"],
        solver_chain=solver_chain,
        model=os.getenv("MODEL_OPTIMIZER"),
        n_variants=int(os.getenv("N_VARIANTS", 5))
    )

    utils.saving_data(
        variant_response=variant_response,
        dataset=os.getenv("DATASET_NAME"),
        question=problem["question"],
        model_optimizer=os.getenv("MODEL_OPTIMIZER"),
        model_target=os.getenv("MODEL_TARGET"),
        difficulty=problem["difficulty"],
        input_output=problem["input_output"],
        baseline_code=baseline_code,
        algorithm="resampling"
    )