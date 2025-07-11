from generator.rules import check_rules
from generator.rules_auto_z3 import check_rules_z3
from utils.z3_utils import instantiate_args, create_z3_args, initial_constraints, collect_constraints
from .inputs import get_inputs
from utils.new_api_utils import get_n_variations, get_lib_version, get_signature, get_api_suffix
from utils.misc import get_dir_in_root, get_tmp_dir, create_subdir
from generator.input_generators import abstract_print, get_abstract_input
from eval.oracle import oracle_crash
import os, sys
import time

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

def reduce_ruleset(ruleset, signature, api, z3_args, max_trial=30, time_budget=30, print_details=False, lib="torch"):
    """
    If after removing a rule, all generated inputs are still valid,
    then the rule is filtered out from the ruleset.
    This stage run until max_trial trials for each rule OR time_budget seconds,
    whichever comes first.
    """
    use_reference = False   # no reduction for reference rulesets
    rules_to_keep = set()
    n_rules_original = len(ruleset)
    base_validity_ratio = 0.0
    base_validity_ratio = 0.0

    time_budget_per_rule = time_budget / (n_rules_original + 1) # Adding 1 for calculating the initial validity ratio
    for rule in [None] + list(ruleset):
        trial = 0
        valid = 0
        valid = 0
        block_all = set()
        perma_block = set()

        start_time = time.time()
        while time.time() - start_time < time_budget_per_rule and trial < max_trial:
            block_one = []
            remaining_ruleset = set(ruleset)
            if rule is not None: 
                remaining_ruleset.remove(rule)

            solver = Solver()
            initial_constraints(solver, signature, z3_args, lib=lib)
            collect_constraints(solver, api, remaining_ruleset, z3_args, use_reference=use_reference)
            # collect_neg_constraint(solver, api, rule, z3_args, use_reference=use_reference)
    
            sampled_blocks = random.sample(list(block_all), int(len(block_all) * 0.3))
            solver.add(*sampled_blocks)
            solver.add(*perma_block)

            if solver.check() != sat:
                trial += 1
                continue
                trial += 1
                continue
    
            model = solver.model()
            for decl in model.decls():
                if decl.arity() != 0:
                    continue
                var, val = decl(), model[decl]
                name_parts = str(decl.name()).rsplit("_", 1)

                if len(name_parts) == 2:
                    prefix, suffix = name_parts
                else:
                    prefix, suffix = name_parts[0], None

                if val.sort().kind() == Z3_ARRAY_SORT:                
                    array_len = None
                    if suffix == "shape" or suffix == "values":
                        for other_decl in model.decls():
                            if str(other_decl.name()) in [f"{prefix}_ndim", f"{prefix}_length"]:
                                array_len = model.eval(other_decl(), model_completion=True).as_long()
                        if array_len is None:
                            array_len = MAX_N_DIM
                    elif suffix == "range":
                        array_len = 2
                    for i in range(array_len):
                        block_one.append(Select(var, i) != model.eval(Select(var, i), model_completion=True))
                    
                    # Do not dim_size to be 0 more than once for a dimension in the shape
                    if model.eval(Select(var, i), model_completion=True).as_long() == 0 and suffix == "shape":
                        perma_block.add(Select(var, i) != model.eval(Select(var, i), model_completion=True))
                else:
                    block_one.append(var != val)
                    
                    if suffix == "ndim" and val.as_long() == 0:
                        perma_block.add(var != val)
    
            for elem in block_one:
                if elem not in block_all:
                    block_all.add(elem)
    
            concrete_input, abstract_input = instantiate_args(model, signature, z3_args, lib=lib)
            status, exception_message = oracle_crash(api, concrete_input, cpu=True, lib=lib)
            
            if status != "invalid":
                valid += 1

            trial += 1

        if rule is None:
            base_validity_ratio = valid / trial
        elif valid / trial < base_validity_ratio:
            rules_to_keep.add(rule)

    print(f"{bcolors.OKBLUE}Rules reduced from {n_rules_original} to {len(rules_to_keep)}{bcolors.ENDC}")
    if print_details and rules_to_keep:
        print(f"Refined rules for {api}:")
        for arity, rule_name, *args in rules_to_keep:
            print(f"- {rule_name} with arity {arity} on args {args}")

    return rules_to_keep


def infer_invariants(api, print_details=False, regen=False, lib="torch", time_budget=60, min_val_inp=100, seed=42, z3=False, suffix=0, use_reference=False):
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

    # Doing a 25/75 split of the time budget for generating inputs and refining rules
    time_budget_learner, time_budget_refinement = 0.25*time_budget, 0.75*time_budget
    
    n_variants = get_n_variations(api, lib=lib)
    if suffix > 0 or (suffix == 0 and n_variants == 1):
        variants = [(get_lib_version(api, lib=lib), suffix)]
    else:
        variants = [(get_lib_version(api, lib=lib), i) for i in range(1, n_variants + 1)]

    for api, suff in variants:
        variant = f"{api}_{suff}" if suff > 0 else api
        invariant_file = os.path.join(get_dir_in_root(f"invariants_{lib}"), variant) if not use_reference else os.path.join(get_dir_in_root(f"reference_invariants_{lib}"), variant)
        # Unless regeneration is forced, return existing ruleset
        if os.path.isfile(invariant_file) and not regen:
            ruleset = read_invariants(invariant_file)
        elif use_reference:
            print(f"No reference invariants found for {variant}. Skipping inference.")
            continue
        else:   # Inference
            print(f"Started invariant inference for {api} (suffix {suff})")
            start_time = time.time()
            if os.path.isfile(invariant_file):
                print(f"Removing existing invariants file for {variant} at {invariant_file}")
                os.remove(invariant_file)
            list_of_inputs = get_inputs(api, lib=lib, time_budget=time_budget_learner, min_val_inp=min_val_inp, seed=seed, suffix=suff)
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
            invalid_inputs = []

            # Verifying stage: Keep rules that ALL valid inputs satisfy
            for idx, input_dict in enumerate(list_of_inputs):
                if print_details:
                    print(f"\n[Input {idx}]")
                    try:
                        print(abstract_print(get_abstract_input(input_dict, api_signature), api_signature))
                    except Exception as e:
                        print(f"Error printing abstract input for {variant}.\n{e.__class__.__name__}: {e}")
                        
                status, exception_message = oracle_crash(api, input_dict, cpu=True, lib=lib)
                if status == "invalid":
                    invalid += 1
                    if print_details:
                        print(f"Input {idx} is invalid")
                        print(f"Exception: {exception_message}")
                        invalid_inputs.append(input_dict)
                else:
                    if print_details:
                        print(f"Input {idx} is valid")
                    # Check rules for the input dictionary
                    if not initialized:
                        ruleset = check_rules_z3(api, input_dict, lib=lib) if z3 else check_rules(input_dict)
                        initialized = True
                    else:
                        ruleset = ruleset.intersection(check_rules_z3(api, input_dict, lib=lib) if z3 else check_rules(input_dict))
                    valid += 1
            print(f"Invariant inference took {time.time()-start_time:.2f} seconds\n")

            # Refining stage: If removing a rule does not decrease the validity ratio, remove it
            print(f"Started rule refinement stage for api {api} (suffix {suff})")
            start_time = time.time()
            z3_args = create_z3_args(api_signature)
            ruleset = reduce_ruleset(ruleset, api_signature, api, z3_args, max_trial=min_val_inp, time_budget=time_budget_refinement, print_details=print_details, lib=lib)
            print(f"Rule refinement took {time.time()-start_time:.2f} seconds")

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
    # Usage: python -m learner.invariant_inference <variant> <time budget> <1 to regenerate invariants 0 otherwise>
    variant = sys.argv[1] if len(sys.argv) > 1 else "scatter"
    budget = int(sys.argv[2]) if len(sys.argv) > 2 else 30  # seconds
    regen = int(sys.argv[3]) == 1 if len(sys.argv) > 3 else False
    lib = sys.argv[4] if len(sys.argv) > 4 else "torch"
    
    api, suffix = get_api_suffix(variant)
    list_of_rulesets = infer_invariants(api, print_details=True, regen=regen, time_budget=budget, z3=True, lib=lib, suffix=suffix)
    
if __name__ == "__main__":
    main()
