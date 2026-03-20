from dotenv import load_dotenv
import os
import utils
from tqdm import tqdm
from APO import automatic_prompt_engineering

load_dotenv()

train_data, _ = utils.load_apps_dataset()

solver_chain = utils.build_solver_chain(model=os.getenv("MODEL_TARGET"))

samples = utils.get_minibatch(
    train_data,
    n_per_difficulty=int(os.getenv("N_PER_DIFFICULTY", 50))
)


#Loop resampling
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


# Loop Automatic Prompt Engineering
CRITIQUE_SAMPLE_EVERY = 5
for i, problem in enumerate(tqdm(samples, desc="Problemi processati")):

    baseline_code = utils.extract_code(
        solver_chain.invoke({"user_prompt": problem["question"]})
    )

    variant_response, critique = automatic_prompt_engineering(
        user_prompt=problem["question"],
        baseline_code=baseline_code,
        target_model=os.getenv("MODEL_TARGET"),
        socratic_model=os.getenv("MODEL_OPTIMIZER"),
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
        algorithm="APO",
        critique=critique if i % CRITIQUE_SAMPLE_EVERY == 0 else None
    )