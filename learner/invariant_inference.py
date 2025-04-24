from generator.rules import check_rules
from .inputs import get_inputs
from utils.api_utils import get_driver
from utils.misc import get_dir_in_root
import os, sys

def save_invariants(api, ruleset, invariant_file):
    if len(ruleset) > 0:
        # If there are rules that have been passed, write them
        with open(invariant_file, "w") as fi:
            for rule in sorted(list(ruleset)):
                fi.write(f"{api},{','.join(list(rule))}\n")

def read_invariants(invariant_file):
    ruleset = set()
    with open(invariant_file, "r") as f:
        for line in f.readlines():
            ruleset.add(tuple(line.strip().split(',')[1:]))
    return ruleset

def print_rules(api, ruleset):
    if len(ruleset) > 0:
        print(f"Rules passed for {api}:")
        for rule, arg1, arg2 in ruleset:
            print(f"- {rule} between {arg1} and {arg2}")
    else:
        print(f"No rules passed for {api}.")

def infer_invariants(api, print_details=False, regen=False, lib="torch", time_budget=30, min_val_inp=5, seed=42):
    '''
        Takes an API and
        
        - returns the invariants if invariants are saved
        - generates inputs otherwise
        
        Inputs are generated randomly using the time_budget
        and seed passed to the API.
        
        If regen is passed as True, forces inferring invariants
        again even if they are saved.
    '''
    invariant_file = os.path.join(get_dir_in_root("invariants"), api)
    # Unlese regeneration is forced, return existing ruleset
    if os.path.isfile(invariant_file) and not regen:
        ruleset = read_invariants(invariant_file)
    else:   # Inference
        list_of_inputs = get_inputs(api, lib=lib, time_budget=time_budget, min_val_inp=min_val_inp, seed=seed)
        ruleset = set()
        initialized = False
        print(f"Inferring invariants for {api} with {len(list_of_inputs)} inputs\n")
        for idx, input_dict in enumerate(list_of_inputs):
            try:
                out_cpu = get_driver(api)(input_dict, cpu=True)
                if print_details:
                    print(f"Input {idx} is valid")
                # Check rules for the input dictionary
                if not initialized:  # If ruleset is not initialized
                    ruleset = check_rules(input_dict)
                    initialized = True
                else:
                    ruleset = ruleset.intersection(check_rules(input_dict))
            except:
                if print_details:
                    print(f"Input {idx} is invalid")   
        
        save_invariants(api, ruleset, invariant_file)
    
    if print_details:
        print_rules(api, ruleset)
    
    return ruleset

def main():
    # Usage: python -m learner.invariant_inference <api> <time budget> <1 to regenerate invariants 0 otherwise>
    api = sys.argv[1] if len(sys.argv) > 1 else "scatter"
    budget = int(sys.argv[2]) if len(sys.argv) > 2 else 30  # seconds
    regen = int(sys.argv[3]) == 1 if len(sys.argv) > 3 else False
        
    ruleset = infer_invariants(api, print_details=True, regen=regen, time_budget=budget)
    
if __name__ == "__main__":
    main()