import time
import numpy as np
import copy
import os
import pickle
import sys
import logging

from z3 import *
from .definitions import get_definition
from .z3 import create_z3_args, instantiate_args, load_existing_models 
from utils.new_api_utils import get_n_variations, get_lib_version
from utils.misc import create_subdir, get_tmp_dir, get_dir_in_root
from generator.input_generators import abstract_print
from eval.oracle import oracle_crash

logger = logging.getLogger(__name__)

def save_state(api, n_models, nominal, invalid, crash, excp, generated_inputs, tmp_results, input_dir, lib="torch"):
    total = nominal + invalid + crash + excp
    valid_prcnt = round((total-invalid)*100/total,2) if total > 0 else 0
    # Save outputs
    csv_file = os.path.join(tmp_results, f"{api}_{lib}.csv")
    with open(csv_file, "w") as f:
        # api, n_models, nominal, invalid, crash, excp, total, valid_prcnt
        f.write(f"{api},{n_models},{nominal},{invalid},{crash},{excp},{total},{valid_prcnt}\n")
    # Save generated inputs
    with open(os.path.join(input_dir, f"{api}_{lib}_inputs.pkl"), "wb") as f_in:
        pickle.dump(generated_inputs, f_in)

def run_api_with_duration(api, duration, n_max=0, seed=42, lib="torch", print_details=False, use_reference=False):
    api = get_lib_version(api, lib=lib)

    # Initialize directories
    input_dir = create_subdir(get_tmp_dir(), "fuzz_inputs")
    tmp_results = create_subdir(get_tmp_dir(), "fuzz_results")
    log_dir = create_subdir(get_tmp_dir(), "fuzz_logs")
    logfile = os.path.join(log_dir, f"{api}.log")

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,                                     # Minimum log level
        format='%(asctime)s - %(levelname)s - %(message)s',     # Log format
        filename=logfile,                                       # Log file path
        filemode="w"                                            # Append/Write mode
    )
    
    print(f"Fuzzing {api} with a {duration} second budget using {lib} library.")
    execution_time = 0
    elapsed = 0
    last_saved = 0
    save_interval = 60 # seconds, 1 minute
    nominal = 0
    invalid = 0
    crash = 0
    excp = 0
    total = 0
    generated_inputs = []
    
    corpus_dir = "corpus_tf" if lib == "tf" else "corpus_torch"

    n_variations = get_n_variations(api, lib=lib)
    model_collection = {}
    if n_variations > 1:
        for i in range(1, n_variations + 1):
            model_collection[i] = {
                "models": []
            }
    else:
        model_collection[0] = {
            "models": []
        }

    for suffix in model_collection.keys():
        model_dir = os.path.join(get_dir_in_root(corpus_dir), f"{api}_{suffix}" if suffix > 0 else api)
        definition = get_definition(api, z3=True, lib=lib, suffix=suffix, use_reference=use_reference)
        if len(definition["ruleset"]) == 0:
            print(f"No invariants learned for {api}_{suffix}. Skipping.")
            continue
        model_collection[suffix]["z3_args"] = create_z3_args(definition["signature"])
        if os.path.exists(corpus_dir):
            model_collection[suffix]["models"] = load_existing_models(model_dir, model_collection[suffix]["z3_args"])
        
        if len(model_collection[suffix]['models']) == 0:
            print(f"No existing models found for {api}_{suffix} in {corpus_dir}. Skipping.")
            continue
        else:
            print(f"Loaded {len(model_collection[suffix]['models'])} existing models for {api}")
    
    # Average number of models across all suffixes
    n_models = np.mean([len(model_collection[suffix]['models']) for suffix in model_collection.keys()])
    rng_model = np.random.default_rng(seed) # random generator for models

    temp_model_collection = copy.deepcopy(model_collection)

    logger.info(f"Starting fuzzing {api} with {n_models} models on average across {len(model_collection.keys())} signature variations.")
    start = time.time()
    while elapsed < duration:
        if elapsed - last_saved > save_interval:
            save_state(api, n_models, nominal, invalid, crash, excp, generated_inputs, tmp_results, input_dir, lib=lib)
            last_saved = elapsed

        seed += 1

        if len(model_collection.keys()) == 0:
            print(f"No models available for {api}. Exiting.")
            break

        suffix_index = rng_model.integers(len(temp_model_collection.keys()))
        selected_suffix = list(temp_model_collection.keys())[suffix_index]
        if len(model_collection[selected_suffix]['models']) == 0:
            print(f"No models generated for {api}_{selected_suffix}. Skipping.")
            # Removing the suffix from the model collection
            model_collection.pop(selected_suffix, None)
            temp_model_collection.pop(selected_suffix, None)
            continue
        
        # Select a variation of the API (e.g. a different signature) at random
        selected_model = rng_model.integers(len(temp_model_collection[selected_suffix]['models']))
        model = temp_model_collection[selected_suffix]['models'][selected_model]
        definition = get_definition(api, z3=True, lib=lib, suffix=selected_suffix, use_reference=use_reference)
        
        concrete_input, abstract_input = instantiate_args(model, definition["signature"], model_collection[selected_suffix]['z3_args'], seed=seed)
        generated_inputs.append((0, abstract_input, seed, selected_suffix))  # first element is distance, set as 0 for consistency
        total += 1
        
        # Print the abstract input if print_details is True
        if print_details:
            print(f"\nAbstract input (seed {seed}):\n{abstract_print(abstract_input, definition['signature'])}")

        start_execution = time.time()
        status, exception_message = oracle_crash(api, concrete_input, cpu=True, lib=lib)
        if status == "nominal":
            nominal += 1
            if print_details:
                print(f"\nNominal input")
        elif status == "invalid":
            invalid += 1
            ## Traceback for debugging
            if print_details:
                print(f"\nThe input might be invalid. Faced exception:\n{exception_message}")
        elif status.endswith("_excp"):
            excp += 1
            # Always log crashes
            logger.error(f"Status: {status}, Exception: {exception_message}, Signature suffix: {selected_suffix}\nInput (seed {seed}):\n{abstract_print(abstract_input, definition['signature'])}")
            print(f"\n[{status}]\n{exception_message}")
            if not print_details:   # if print_details is True, the abstract input is already printed
                print(f"\nAbstract input (seed {seed}):\n{abstract_print(abstract_input, definition['signature'])}")
        elif status.endswith("_crash"):
            crash += 1
            # Always log crashes
            logger.error(f"Status: {status}, Exception: {exception_message}, Signature suffix: {selected_suffix}\nInput (seed {seed}):\n{abstract_print(abstract_input, definition['signature'])}")
            print(f"\n[{status}]\n{exception_message}")
            if not print_details:   # if print_details is True, the abstract input is already printed
                print(f"\nAbstract input (seed {seed}):\n{abstract_print(abstract_input, definition['signature'])}")
        else:
            logger.error(f"Status: {status}, Exception: {exception_message}, Signature suffix: {selected_suffix}\nInput (seed {seed}):\n{abstract_print(abstract_input, definition['signature'])}")
            if print_details:
                print(f"\nThe input faced status {status}. Faced exception:\n{exception_message}")
        
        execution_time = execution_time + time.time() - start_execution
        print_str = f"Nominal: {nominal} | Invalid: {invalid} | Crash: {crash} | Exception: {excp} | Last saved: {round(elapsed-last_saved, 2)}s ago"
        
        if print_details:
            print(print_str)            
        else:
            print(print_str, end='\r', flush=True)

        # If n_max is defined and n_max inputs have been generated, exit
        if n_max > 0 and total >= n_max:
            break

        elapsed = time.time() - start

        # Don't reuse the same model until all models have been used
        del temp_model_collection[selected_suffix]['models'][selected_model]
        if len(temp_model_collection[selected_suffix]['models']) == 0:
            temp_model_collection[selected_suffix]['models'] = copy.deepcopy(model_collection[suffix]['models'])

    logger.info(f"Fuzzing completed for {api}. Total inputs: {total}, Nominal: {nominal}, Invalid: {invalid}, Crash: {crash}, Exception: {excp}.")
    total_time = time.time() - start
    valid_prcnt = round((total-invalid)*100/total,2) if total > 0 else 0
    print(f"\n[{api}]\n\tOptimzation took {round(total_time-execution_time, 4)}s\n\tExecuting {nominal+invalid} inputs on {api} took {round(execution_time, 4)}s\n\tTotal {round(total_time, 4)}s")
    print(f"Models (average): {n_models} | Nominal: {nominal} | Invalid: {invalid} | Crash: {crash} | Exception: {excp} | Total {total} | Validity Rate: {valid_prcnt}%")
    
    save_state(api, n_models, nominal, invalid, crash, excp, generated_inputs, tmp_results, input_dir, lib=lib)
        

if __name__ == "__main__":
    seed = 200
    fuzz_duration = 30 # seconds
    lib = "torch"
    print_details = sys.argv[1].lower() == 'true' if len(sys.argv) > 1 else False
    
    run_api_with_duration("add", fuzz_duration, print_details=print_details, lib=lib)
    run_api_with_duration("combinations", fuzz_duration, print_details=print_details, lib=lib)
