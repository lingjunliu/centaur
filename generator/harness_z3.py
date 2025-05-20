import time
import numpy as np
import traceback
import os
import pickle
import sys
import json

from z3 import *
from .definitions import map_defs, get_definition
from .serialize import load_model, save_model 
from .z3 import create_z3_args, gen_models, instantiate_args 
from utils.api_utils import get_driver
from utils.misc import create_subdir, get_tmp_dir
from generator.input_generators import abstract_print
from eval.oracle import oracle_crash

def save_state(api, valid, invalid, crash, excp, generated_inputs, tmp_results, input_dir, lib="torch"):
    total = valid + invalid + crash + excp
    valid_prcnt = round((total-invalid)*100/total,2) if total > 0 else 0
    # Save outputs
    csv_file = os.path.join(tmp_results, f"{api}_{lib}.csv")
    with open(csv_file, "w") as f:
        # api, valid, invalid, crash, excp, total, valid_prcnt
        f.write(f"{api},{valid},{invalid},{crash},{excp},{total},{valid_prcnt}\n")
    # Save generated inputs
    with open(os.path.join(input_dir, f"{api}_{lib}_inputs.pkl"), "wb") as f_in:
        pickle.dump(generated_inputs, f_in)

def run_api_with_duration(api, model_gen_duration, fuzz_duration, max_model, n_max=0, print_details=False, model_regen=False, seed=42, lib="torch"):
    # Initialize directories
    input_dir = create_subdir(get_tmp_dir(), "fuzz_inputs")
    tmp_results = create_subdir(get_tmp_dir(), "fuzz_results")
    
    # Library specific
    driver = get_driver(api, lib=lib)
    corpus_dir = "corpus_tf" if lib == "tf" else "corpus_torch"

    print(f"Optimizing for {api} with {model_gen_duration} (max_model) and {fuzz_duration} (fuzz) second budgets")
    execution_time = 0
    elapsed = 0
    last_saved = 0
    save_interval = 600 # seconds, 10 minutes
    valid = 0
    invalid = 0
    crash = 0
    excp = 0
    generated_inputs = []
    definition = get_definition(api, z3=True, lib=lib)
    if len(definition["ruleset"]) == 0:
        print(f"No invariants learned for {api}")
        return

    z3_args = create_z3_args(definition["signature"])
    if os.path.exists(f"{corpus_dir}/{api}") and not model_regen:
        models = []
        for model_file in sorted(os.listdir(f"{corpus_dir}/{api}")):
            model_path = os.path.join(f"{corpus_dir}/{api}", model_file)
            with open(model_path, "r") as f:
                model_data = json.load(f)
            model = load_model(model_data, z3_args)
            models.append(model)
        print(f"Loaded {len(models)} existing models for {api}")
    else:
        models = gen_models(definition, driver, z3_args, model_gen_duration, max_model, seed=seed, print_details=print_details)
        os.makedirs(corpus_dir, exist_ok=True)
        os.makedirs(f"{corpus_dir}/{api}", exist_ok=True)
        for idx, model in enumerate(models):
            path = os.path.join(f"{corpus_dir}/{api}", f"model-{idx}.json")
            save_model(model, path)
        print(f"Generated {len(models)} models for {api}")
    
    rng_model = np.random.default_rng(seed) # random generator for models

    start = time.time()
    while len(models) > 0 and elapsed < fuzz_duration:
        if elapsed - last_saved > save_interval:
            save_state(api, valid, invalid, crash, excp, generated_inputs, tmp_results, input_dir, lib=lib)
            last_saved = elapsed

        seed += 1
        model = models[rng_model.integers(len(models))]
        concrete_input, abstract_input = instantiate_args(model, definition["signature"], z3_args, seed=seed)
        generated_inputs.append((0, abstract_input, seed))  # first element is distance, set as 0 for consistency

        start_execution = time.time()
        status, exception_message = oracle_crash(driver, concrete_input, cpu=True)
        if status == "nominal":
            valid += 1
            if print_details:
                print(f"\nNominal input:\n{abstract_print(abstract_input, definition['signature'])}")
        elif status == "invalid":
            invalid += 1
            ## Traceback for debugging
            if print_details:
                print(f"\nThe input might be invalid. Faced exception:\n{exception_message}")
                print(f"\Invalid input:\n{abstract_print(abstract_input, definition['signature'])}")
        elif status.endswith("_excp"):
            excp += 1
            # Always log crashes
            print(f"\n[{status}]\n{exception_message}")
            print(f"\nAbstract input (seed {seed}):\n{abstract_print(abstract_input, definition['signature'])}")
        elif status.endswith("_crash"):
            crash += 1
            # Always log crashes
            print(f"\n[{status}]\n{exception_message}")
            print(f"\nAbstract input (seed {seed}):\n{abstract_print(abstract_input, definition['signature'])}")
        else:
            if print_details:
                print(f"\nThe input faced status {status}. Faced exception:\n{exception_message}")
        execution_time = execution_time + time.time() - start_execution
        print(f"Valid: {valid} | Invalid: {invalid} | Crash: {crash} | Exception: {excp} | Last saved: {round(elapsed-last_saved, 2)}s ago", end='\r', flush=True)

        # If n_max is defined and n_max inputs have been generated, exit
        if n_max > 0 and (valid+invalid) == n_max:
            break

        elapsed = time.time() - start

    total_time = time.time() - start
    total = valid + invalid + crash + excp
    valid_prcnt = round((total-invalid)*100/total,2) if total > 0 else 0
    print(f"\n[{api}]\n\tOptimzation took {round(total_time-execution_time, 4)}s\n\tExecuting {valid+invalid} inputs on {api} took {round(execution_time, 4)}s\n\tTotal {round(total_time, 4)}s")
    print(f"Models: {len(models)} | Valid: {valid} | Invalid: {invalid} | Crash: {crash} | Exception: {excp} | Total {total} | Validity Rate: {valid_prcnt}%")
    
    save_state(api, valid, invalid, crash, excp, generated_inputs, tmp_results, input_dir, lib=lib)
        

if __name__ == "__main__":
    seed = 200
    model_gen_duration = 60 # seconds
    fuzz_duration = 30 # seconds
    max_model = 100
    lib = "torch"
    print_details = sys.argv[1].lower() == 'true' if len(sys.argv) > 1 else False
    
    run_api_with_duration("add", model_gen_duration, fuzz_duration, max_model, print_details=print_details, lib=lib)
    run_api_with_duration("combinations", model_gen_duration, fuzz_duration, max_model, print_details=print_details, lib=lib)
