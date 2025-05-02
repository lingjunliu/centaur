import time
import numpy as np
import traceback
import os
import pickle
import sys

from .ea import Configuration, Mutator, optimize
from .definitions import map_defs, get_definition
from .input_generators import get_random_input
from utils.api_utils import get_driver
from utils.misc import create_subdir, get_tmp_dir

def main():
    # Rules
    seed = 42
    for rule_name in ['rule_1','rule_2','rule_3','rule_4','rule_5','rule_6', 'rule_11']:
        start = time.time()
        # Set a random seed for reproducibility
        seed += 1
        config = Configuration(map_defs[rule_name], seed) # Rule configuration
        mutator = Mutator(config)
        (best_distance, best_input) = optimize(config, mutator)
        print(f"\n{' '.join(rule_name.capitalize().split('_'))}:\nBest Distance: {best_distance}\nBest Input:")
        best_input_abstracted = config.translate_to_input_dict(best_input, abstract=True)
        for key, value in best_input_abstracted.items():
            print(f"\t{key}: {value}")
        print(f"Optimized in {round(time.time()-start, 4)}s")
    
    
    # # Scatter
    # start = time.time()
    # seed = 200
    # map_defs["scatter"]["random_candidate"] =    {
    #                                                 "input": np.random.rand(2,4).astype(np.float32),
    #                                                 "dim": 7,
    #                                                 "index": np.random.rand(2,1).astype(np.int64),
    #                                                 "src": np.random.rand(2,1,3).astype(np.complex128)
    #                                             }
    # config = Configuration(map_defs["scatter"], seed) # Rule 2 and 3 configuration
    # mutator = Mutator(config)
    # (best_distance, best_input) = optimize(config, mutator)
    # print(f"\nScatter:\nBest Distance: {best_distance}\nBest Input:")
    # best_input_abstracted = config.translate_to_input_dict(best_input, abstract=True)
    # for key, value in best_input_abstracted.items():
    #     print(f"\t{key}: {value}")
    # print(f"Optimized in {round(time.time()-start, 4)}s")
    
    # # verify if the input is valid
    # try:
    #     scatter_cpu = get_driver("scatter")(config.translate_to_input_dict(best_input), cpu=True)
    #     print("\nThe input for scatter was valid! Yayyy!!!")
    # except Exception as e:
    #     print(f"\nThe input might be invalid. Faced exception:\n{e.__class__}: {str(e)}\n\n")
        
    #     ## Traceback for debugging
    #     traceback.print_exc()
    
def run_api_with_duration(api, duration, n_max=0, limit=30, print_details=False):
    driver = get_driver(api)

    print(f"Optimizing for {api} with a {duration} second budget")
    execution_time = 0
    start = time.time()
    elapsed = 0
    valid = 0
    invalid = 0
    seed = 200
    generated_inputs = []
    definition = get_definition(api)
    if len(definition["ruleset"]) == 0:
        print(f"No invariants learned for {api}")
        return
    while elapsed < duration:
        seed += 1
        config = Configuration(definition, seed)
        # TODO: Debug why initializing random candidate makes optimizer slow
        config.random_candidate = get_random_input(definition["signature"], config.rng)
        # config.set_random_candidate(map_defs[api]["random_candidate"])
        mutator = Mutator(config)
        (best_distance, best_input) = optimize(config, mutator, duration=limit)
        # Save abstract versions of the inputs with seed for reproduction
        generated_inputs.append((best_distance, best_input, seed))
        if print_details:
            print(f"\n{api}:\nBest Distance: {best_distance}\nBest Input:")
            best_input_abstracted = config.translate_to_input_dict(best_input, abstract=True)
            for key, value in best_input_abstracted.items():
                print(f"\t{key}: {value}")
        
        # verify if the input is valid
        start_execution = time.time()
        try:
            out_cpu = driver(config.translate_to_input_dict(best_input, seed=seed), cpu=True)
            valid += 1
        except Exception as e:
            invalid += 1
            ## Traceback for debugging
            if print_details:
                print(f"\nThe input might be invalid. Faced exception:\n{e.__class__}: {str(e)}")
                traceback.print_exc()
        execution_time = execution_time + time.time() - start_execution
        print(f"Valid: {valid} | Invalid: {invalid}", end='\r', flush=True)
        
        # If n_max is defined and n_max inputs have been generated, exit
        if n_max > 0 and (valid+invalid) == n_max:
            break
        
        elapsed = time.time() - start
    
    total_time = time.time() - start
    valid_prcnt = round(valid*100/(valid+invalid),2) if valid+invalid > 0 else 0
    print(f"\n[{api}]\n\tOptimzation took {round(total_time-execution_time, 4)}s\n\tExecuting {valid+invalid} inputs on {api} took {round(execution_time, 4)}s\n\tTotal {round(total_time, 4)}s")
    print(f"Valid: {valid} | Invalid: {invalid} | Total {valid+invalid} | Validity Rate: {valid_prcnt}%")
    
    # Save outputs
    tmp_results = create_subdir(get_tmp_dir(), "fuzz_results")
    csv_file = os.path.join(tmp_results, f"{api}_{duration}.csv")
    with open(csv_file, "w") as f:
        f.write(f"{api},{valid},{invalid},{valid_prcnt}\n")
    input_dir = create_subdir(get_tmp_dir(), "fuzz_inputs")
    with open(os.path.join(input_dir, f"{api}_inputs.pkl"), "wb") as f_in:
        pickle.dump(generated_inputs, f_in)

if __name__ == "__main__":
    main()
    # Run scatter for 30 minutes
    duration = 30 # seconds
    limit = 10  # random restart after <limit> seconds
    print_details = int(sys.argv[1]) == 1 if len(sys.argv) > 1 else False
    
    run_api_with_duration("scatter", duration, print_details=print_details, limit=limit)
    
    # Run atan2 for 30 seconds
    run_api_with_duration("atan2", duration, print_details=print_details, limit=limit)
    
    # Run argmin for 30 seconds
    run_api_with_duration("argmin", duration, print_details=print_details, limit=limit)
    
    # Run conv_transpose2d for 30 seconds
    run_api_with_duration("conv_transpose2d", duration, print_details=print_details, limit=limit)