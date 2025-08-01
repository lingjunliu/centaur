import time
import numpy as np
import copy
import os
import pickle
import sys

from z3 import *
from .z3 import load_existing_models, load_abstract_inputs
from utils.z3_utils import instantiate_args, create_z3_args
from utils.new_api_utils import get_n_variations, get_lib_version, get_signature
from utils.misc import create_subdir, get_tmp_dir, get_dir_in_root
from generator.input_generators import abstract_print, concretize_input
from eval.oracle import oracle_crash

import logging

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

def run_api_with_duration(api, duration, n_max=0, seed=42, lib="torch", print_details=False, use_reference=False, use_abstracts=True):
    api = get_lib_version(api, lib=lib)

    # Initialize directories
    input_dir = create_subdir(get_tmp_dir(), "fuzz_inputs")
    tmp_results = create_subdir(get_tmp_dir(), "fuzz_results")
    log_dir = create_subdir(get_tmp_dir(), "fuzz_logs")
    logfile = os.path.join(log_dir, f"{api}.log")

    logger = logging.getLogger(__name__)
    # Configure logging for this specific logger
    logger.setLevel(logging.INFO)
    
    # Create file handler
    file_handler = logging.FileHandler(logfile, mode="w")
    file_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(file_handler)
    
    print(f"Fuzzing {api} with a {duration} second budget using {lib} library.")
    print(f"Logging details to {logfile}")
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
        cur_sig = get_signature(api, lib=lib, suffix=suffix)
        model_collection[suffix]["z3_args"] = create_z3_args(cur_sig)
        if os.path.exists(model_dir):
            model_collection[suffix]["models"] = load_abstract_inputs(model_dir, cur_sig, lib=lib) if use_abstracts else load_existing_models(model_dir, model_collection[suffix]["z3_args"])
        else:
            print(f"No existing models directory found for {api}_{suffix} (expected {model_dir}). Skipping.")
            continue

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
        cur_sig = get_signature(api, lib=lib, suffix=selected_suffix)
        
        if use_abstracts:
            abstract_input = model
            concrete_input = concretize_input(abstract_input, cur_sig, rng=np.random.default_rng(seed))
        else:
            concrete_input, abstract_input = instantiate_args(model, cur_sig, model_collection[selected_suffix]['z3_args'], seed=seed, lib=lib)

        generated_inputs.append((0, abstract_input, seed, selected_suffix))  # first element is distance, set as 0 for consistency
        
        # Print the abstract input if print_details is True
        abstract_str = f"[{total}] Abstract input (seed {seed}, suffix: {selected_suffix})\n{abstract_print(abstract_input, cur_sig)}"
        logger.info(abstract_str)
        if print_details:
            print(f"\n{abstract_str}")
        
        total += 1
        log_func = logger.info

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
            log_func = logger.error
            print(f"\n[{status}]\n{exception_message}")
            if not print_details:   # if print_details is True, the abstract input is already printed
                print(f"\nAbstract input (seed {seed}):\n{abstract_print(abstract_input, cur_sig)}")
        elif status.endswith("_crash"):
            crash += 1
            # Always log crashes
            log_func = logger.error
            print(f"\n[{status}]\n{exception_message}")
            if not print_details:   # if print_details is True, the abstract input is already printed
                print(f"\nAbstract input (seed {seed}):\n{abstract_print(abstract_input, cur_sig)}")
        else:
            if print_details:
                print(f"\nThe input faced status {status}. Faced exception:\n{exception_message}")
        
        log_func(f"Status: {status}, Exception: {exception_message}") if exception_message else log_func(f"Status: {status}")
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
    print(f"Models (average): {n_models} | Nominal: {nominal} | Invalid: {invalid} | Crash: {crash} | Exception: {excp} | Total {total} | Validity Rate: {valid_prcnt}%")
    
    save_state(api, n_models, nominal, invalid, crash, excp, generated_inputs, tmp_results, input_dir, lib=lib)
        

def main():
    if len(sys.argv) < 3:
        print("Usage: python -m generator.harness_z3 <api> <duration> <n_max, optional> <lib, default: torch> <seed, optional> <print_details, optional>")
        return
    
    api = sys.argv[1]
    duration = int(sys.argv[2])
    n_max = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    lib = sys.argv[4] if len(sys.argv) > 4 else "torch"
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 200
    print_details = sys.argv[6].lower() == 'true' if len(sys.argv) > 6 else False
    use_reference = sys.argv[7].lower() == 'true' if len(sys.argv) > 7 else False

    # alias
    if lib == "tensorflow":
        lib = "tf"
    elif lib == "pytorch":
        lib = "torch"
    
    run_api_with_duration(api, duration, n_max=n_max, seed=seed, lib=lib, print_details=print_details, use_reference=use_reference)

if __name__ == "__main__":
    main()
