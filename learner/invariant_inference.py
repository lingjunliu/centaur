from generator.rules import check_rules
from generator.rules_z3 import check_rules_z3
from .inputs import get_inputs
from utils.api_utils import get_driver
from utils.misc import get_dir_in_root
from generator.input_generators import abstract_print, get_abstract_input
from eval.oracle import oracle_crash
import os, sys

def save_invariants(api, ruleset, invariant_file):
    if len(ruleset) > 0:
        # If there are rules that have been passed, write them
        with open(invariant_file, "w") as fi:
            for rule in sorted(list(ruleset)):
                fi.write(f"{api},{rule[0]},{','.join(list(rule[1:]))}\n")

def read_invariants(invariant_file):
    ruleset = set()
    with open(invariant_file, "r") as f:
        for line in f.readlines():
            parts = line.strip().split(',')[1:]
            ruleset.add(tuple([int(parts[0])] + parts[1:]))
    return ruleset

def print_rules(api, ruleset):
    if len(ruleset) > 0:
        print(f"Rules passed for {api}:")
        for arity, rule_name, *args in ruleset:
            print(f"- {rule_name} with arity {arity} on args {args}")
    else:
        print(f"No rules passed for {api}.")

def infer_invariants(api, print_details=False, regen=False, lib="torch", time_budget=30, min_val_inp=5, seed=42, z3=False):
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
            status, exception_message = oracle_crash(api, input_dict, cpu=True)
            if status == "invalid":
                if print_details:
                    print(f"Input {idx} is invalid")
            else:
                if print_details:
                    print(abstract_print(get_abstract_input(input_dict)))
                    print(f"Input {idx} is valid")
                # Check rules for the input dictionary
                if not initialized:  # If ruleset is not initialized
                    ruleset = check_rules_z3(input_dict) if z3 else check_rules(input_dict)
                    initialized = True
                else:
                    ruleset = ruleset.intersection(check_rules_z3(input_dict) if z3 else check_rules(input_dict))             
        
        save_invariants(api, ruleset, invariant_file)
    
    if print_details:
        print_rules(api, ruleset)
    
    return ruleset

def main():
    # Usage: python -m learner.invariant_inference <api> <time budget> <1 to regenerate invariants 0 otherwise>
    api = sys.argv[1] if len(sys.argv) > 1 else "scatter"
    budget = int(sys.argv[2]) if len(sys.argv) > 2 else 30  # seconds
    regen = int(sys.argv[3]) == 1 if len(sys.argv) > 3 else False
        
    ruleset = infer_invariants(api, print_details=True, regen=regen, time_budget=budget, z3=True)
    
if __name__ == "__main__":
    main()
