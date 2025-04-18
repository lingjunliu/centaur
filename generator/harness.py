import time
import numpy as np
import traceback

from .ea import Configuration, Mutator, optimize
from .definitions import map_defs, api_defs

def main():
    # Rules
    seed = 42
    for rule_name in ['rule_1','rule_2','rule_3','rule_4','rule_5','rule_6']:
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
    
    
    # Scatter
    start = time.time()
    seed = 200
    map_defs["scatter"]["random_candidate"] =    {
                                                    "input": np.random.rand(2,4).astype(np.float32),
                                                    "dim": 7,
                                                    "index": np.random.rand(2,1).astype(np.int64),
                                                    "src": np.random.rand(2,1,3).astype(np.complex128)
                                                }
    config = Configuration(map_defs["scatter"], seed) # Rule 2 and 3 configuration
    mutator = Mutator(config)
    (best_distance, best_input) = optimize(config, mutator)
    print(f"\nScatter:\nBest Distance: {best_distance}\nBest Input:")
    best_input_abstracted = config.translate_to_input_dict(best_input, abstract=True)
    for key, value in best_input_abstracted.items():
        print(f"\t{key}: {value}")
    print(f"Optimized in {round(time.time()-start, 4)}s")
    
    # verify if the input is valid
    try:
        scatter_cpu = api_defs["scatter"].torch_version(config.translate_to_input_dict(best_input), cpu=True)
        print("\nThe input for scatter was valid! Yayyy!!!")
    except Exception as e:
        print(f"\nThe input might be invalid. Faced exception:\n{e.__class__}: {str(e)}\n\n")
        
        ## Traceback for debugging
        traceback.print_exc()
    
def run_api_with_duration(api, duration, print_details=False):
    if api not in api_defs:
        print(f"{api} not supported yet")
        return

    print(f"Optimizing for {api} with a {duration} second budget")
    execution_time = 0
    start = time.time()
    valid = 0
    invalid = 0
    seed = 200
    while time.time() - start < duration:
        seed += 1
        config = Configuration(map_defs[api], seed) # api definition
        mutator = Mutator(config)
        (best_distance, best_input) = optimize(config, mutator)
        if print_details:
            print(f"\n{api}:\nBest Distance: {best_distance}\nBest Input:")
            best_input_abstracted = config.translate_to_input_dict(best_input, abstract=True)
            for key, value in best_input_abstracted.items():
                print(f"\t{key}: {value}")
        
        # verify if the input is valid
        start_execution = time.time()
        try:
            out_cpu = api_defs[api].torch_version(config.translate_to_input_dict(best_input), cpu=True)
            print(f"valid: {valid}", end='\r', flush=True)
            valid += 1
        except Exception as e:
            invalid += 1
            ## Traceback for debugging
            if print_details:
                print(f"\nThe input might be invalid. Faced exception:\n{e.__class__}: {str(e)}")
                traceback.print_exc()
        execution_time = execution_time + time.time() - start_execution
    print(f"\nOptimized {api} in {round(time.time()-start, 4)}s | API Execution took {execution_time} seconds")
    print(f"Valid: {valid} | Invalid: {invalid} | Total {valid+invalid} | Validity Rate: {round(valid*100/(valid+invalid),2) if valid+invalid > 0 else 0}%")

if __name__ == "__main__":
    # main()    
    # Run scatter for 30 minutes
    duration = 30 # seconds
    run_api_with_duration("scatter", duration)
    
    # Run atan2 for 30 seconds
    run_api_with_duration("atan2", duration)
    
    # Run argmin for 30 seconds
    run_api_with_duration("argmin", duration)
    
    # Run conv_transpose2d for 30 seconds
    run_api_with_duration("conv_transpose2d", duration)