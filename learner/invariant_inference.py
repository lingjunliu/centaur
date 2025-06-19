from generator.rules import check_rules
from generator.rules_auto_z3 import check_rules_z3
from .inputs import get_inputs
from utils.new_api_utils import get_n_variations, get_lib_version, get_signature
from utils.misc import get_dir_in_root, get_tmp_dir, create_subdir
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

def infer_invariants(api, print_details=False, regen=False, lib="torch", time_budget=30, min_val_inp=20, seed=42, z3=False, suffix=0):
    '''
        Takes an API and
        
        - returns a list of invariants for each possible signature
          of the API if no suffix is passed.
        - returns a list with only one set of invariants for the specific signature
          if a suffix is passed.
        
        Inputs are generated randomly using the time_budget
        if llm generated inputs are not available.
        The seed is passed to the API.
        
        If regen is passed as True or if no invariants exist, inference
        is performed and the invariants are saved to a file. If regen is False
        and the invariants already exist, they are read from the file and returned.
    '''
    list_of_rulesets = []
    
    n_variants = get_n_variations(api, lib=lib)
    if suffix > 0 or (suffix == 0 and n_variants == 1):
        variants = [(get_lib_version(api, lib=lib), suffix)]
    else:
        variants = [(get_lib_version(api, lib=lib), i) for i in range(1, n_variants + 1)]

    for api, suff in variants:
        variant = f"{api}_{suff}" if suff > 0 else api
        invariant_file = os.path.join(get_dir_in_root(f"invariants_{lib}"), variant)
        # Unlese regeneration is forced, return existing ruleset
        if os.path.isfile(invariant_file) and not regen:
            ruleset = read_invariants(invariant_file)
        else:   # Inference
            list_of_inputs = get_inputs(api, lib=lib, time_budget=time_budget, min_val_inp=min_val_inp, seed=seed, suffix=suff)
            ruleset = set()
            initialized = False
            print(f"Inferring invariants for {variant} with {len(list_of_inputs)} inputs\n")
            try:
                api_signature = get_signature(api, lib=lib, suffix=suff)
            except Exception as e:
                print(f"Error getting signature for {variant}.\n{e.__class__.__name__}: {e}")
                continue
            
            valid = 0
            invalid = 0
            for idx, input_dict in enumerate(list_of_inputs):
                status, exception_message = oracle_crash(api, input_dict, cpu=True, lib=lib)
                if status == "invalid":
                    invalid += 1
                    if print_details:
                        print(f"Input {idx} is invalid")
                        print(f"Exception: {exception_message}")
                else:
                    if print_details:
                        print(abstract_print(get_abstract_input(input_dict, api_signature), api_signature))
                        print(f"Input {idx} is valid")
                    # Check rules for the input dictionary
                    if not initialized:  # If ruleset is not initialized
                        ruleset = check_rules_z3(input_dict) if z3 else check_rules(input_dict)
                        initialized = True
                    else:
                        ruleset = ruleset.intersection(check_rules_z3(input_dict) if z3 else check_rules(input_dict))
                    valid += 1
            # Save some stats
            infer_dir = create_subdir(get_tmp_dir(), f"infer_results_{lib}")
            csv_file = os.path.join(infer_dir, f"{variant}.csv")
            with open(csv_file, "w") as f:
                f.write(f"{api},{valid},{invalid},{round(valid*100/(valid+invalid), 4) if (valid+invalid) > 0 else 0}\n")
            save_invariants(api, ruleset, invariant_file)
        
        if print_details:
            print_rules(variant, ruleset)
        
        list_of_rulesets.append(ruleset)

    return list_of_rulesets

def main():
    # Usage: python -m learner.invariant_inference <api> <time budget> <1 to regenerate invariants 0 otherwise>
    api = sys.argv[1] if len(sys.argv) > 1 else "scatter"
    budget = int(sys.argv[2]) if len(sys.argv) > 2 else 30  # seconds
    regen = int(sys.argv[3]) == 1 if len(sys.argv) > 3 else False
    lib = sys.argv[4] if len(sys.argv) > 4 else "torch"
        
    list_of_rulesets = infer_invariants(api, print_details=True, regen=regen, time_budget=budget, z3=True, lib=lib)
    
if __name__ == "__main__":
    main()
