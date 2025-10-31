from generator.rules_auto_z3 import check_rules_z3
from utils.z3_utils import instantiate_args, create_z3_args, initial_constraints, collect_constraints
from .inputs import augment_one_input
from utils.new_api_utils import get_n_variations, get_lib_version, get_signature, get_signature_of_input, get_api_suffix, get_raw_op_mapping
from utils.misc import get_dir_in_root, get_tmp_dir, create_subdir, append_file_in_root, bcolors, read_file_in_root
from utils.defaults import MAX_N_DIM
from generator.input_generators import abstract_print, get_abstract_input, get_random_input
from generator.random_generation import random_fuzz
from eval.oracle import oracle_crash

import os, sys
import time
from z3 import *
import numpy as np

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
    if ruleset is not None and len(ruleset) > 0:
        print(f"Rules passed for {api}:")
        for arity, rule_name, *args in ruleset:
            print(f"- {rule_name} with arity {arity} on args {args}")
    else:
        print(f"No rules passed for {api}.")

def reduce_ruleset(ruleset, signature, api, z3_args, max_trial=30, time_budget=30, print_details=False, lib="torch", rng=np.random.default_rng(42)):
    """
    If after removing a rule, validity ratio does not decrease below the base validity ratio,
    then the rule is filtered out from the ruleset.
    This stage run until max_trial trials for each rule OR time_budget seconds,
    whichever comes first.
    """
    use_reference = False   # no reduction for reference rulesets
    rules_to_keep = set()
    n_rules_original = len(ruleset)
    base_validity_ratio = 0.0
    min_validity_ratio = 0.0
    rule_with_min_validity_ratio = None

    time_budget_per_rule = time_budget / (n_rules_original + 1) # Adding 1 for calculating the initial validity ratio
    for rule in [None] + list(ruleset):
        trial = 0
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
            collect_constraints(solver, api, remaining_ruleset, z3_args, use_reference=use_reference, lib=lib)
            # collect_neg_constraint(solver, api, rule, z3_args, use_reference=use_reference)
    
            sampled_blocks = rng.choice(list(block_all), int(len(block_all) * 0.3), replace=False)
            solver.add(*sampled_blocks)
            solver.add(*perma_block)

            if solver.check() != sat:
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
                    
                        # Do not allow dim_size to be 0 more than once for a dimension in the shape
                        if model.eval(Select(var, i), model_completion=True).as_long() == 0 and suffix == "shape":
                            perma_block.add(Select(var, i) != model.eval(Select(var, i), model_completion=True))
                else:
                    block_one.append(var != val)
                    
                    if suffix == "ndim" and val.as_long() == 0:
                        perma_block.add(var != val)
    
            for elem in block_one:
                if elem not in block_all:
                    block_all.add(elem)
            print(f"Size of block_all: {len(block_all)}, {sys.getsizeof(block_all)*0.001*0.001} MB", flush=True)

            try:
                concrete_input, abstract_input = instantiate_args(model, signature, z3_args, lib=lib, sample_range=False)
            except np.core._exceptions._ArrayMemoryError as e:
                print("Skipping input due to the tensor being too large.")
                continue
            
            status, exception_message = oracle_crash(api, concrete_input, cpu=True, lib=lib)
            
            if status != "invalid":
                valid += 1
            elif print_details and exception_message:
                print(f"[Valid: {valid}, Invalid: {trial + 1 - valid}, Total: {trial + 1}]")
                # try:
                #     print(abstract_print(get_abstract_input(abstract_input, signature), signature))
                # except Exception as e:
                #     print(f"{bcolors.WARNING}Error while printing abstract: {e}{bcolors.ENDC}")
                print(f"Trial {trial} with rule {rule} failed with exception: {exception_message}")

            trial += 1

        cur_validity_ratio = valid / trial if trial > 0 else 0.0
        if rule is None:
            base_validity_ratio = cur_validity_ratio
        elif cur_validity_ratio < base_validity_ratio or base_validity_ratio == 0.0:
            rules_to_keep.add(rule)
        elif print_details:
            if min_validity_ratio == 0.0 or cur_validity_ratio < min_validity_ratio:
                min_validity_ratio = cur_validity_ratio
                rule_with_min_validity_ratio = rule
            print(f"{bcolors.WARNING}Removing rule {rule} did not reduce the validity ratio ({cur_validity_ratio:.4f}) below the base ratio {base_validity_ratio:.4f}. Removing it. Saving minimum validity ratio as {min_validity_ratio:.4f}{bcolors.ENDC}")

    print(f"\n-- Rules reduced from {n_rules_original} to {len(rules_to_keep)} --\n")
    if len(rules_to_keep) == 0:
        print(f"{bcolors.WARNING}No rules passed the reduction stage. Keeping the rule removing which results in the minimum validity ratio.{bcolors.ENDC}")
        rules_to_keep.add(rule_with_min_validity_ratio)

    return rules_to_keep

def get_invariants(api, suffix, lib="torch", use_reference=False):
    variant = f"{api}_{suffix}" if suffix > 0 else api
    invariant_file = os.path.join(get_dir_in_root(f"invariants_{lib}"), variant) if not use_reference else os.path.join(get_dir_in_root(f"reference_invariants_{lib}"), variant)
    if os.path.isfile(invariant_file):
        ruleset = read_invariants(invariant_file)
    else:
        ruleset = set()

    return ruleset

def update_ruleset(api, input_dict, ruleset, lib="torch"):
    status, exception_message = oracle_crash(api, input_dict, cpu=True, lib=lib)
    if status == "nominal":
        api_signature = get_signature_of_input(api, input_dict, lib=lib)
        optional_kwargs = api_signature.get("kwargs", {}).copy()
        optional_kwargs.update({"layout": "", "memory_format": ""})

        optional_none_params = [param for param in optional_kwargs if param in input_dict and input_dict[param] is None]
        for param in optional_none_params:
            del input_dict[param]

        if ruleset == -1:
            ruleset = check_rules_z3(api, input_dict, lib=lib)
        elif ruleset:
            preserved = set()
            for rule in ruleset:
                _, _, *args = rule
                if any(arg in optional_none_params for arg in args):
                    preserved.add(rule)
            ruleset = ruleset.intersection(check_rules_z3(api, input_dict, lib=lib)).union(preserved)

    return ruleset, status, exception_message

def infer_invariants(api, print_details=False, regen=False, lib="torch", time_budget=60, min_val_inp=30, seed=42, z3=True, suffix=0, use_reference=False, reduce_rules=True):
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
    if not z3:
        print(f"{bcolors.FAIL}Only Z3 is supported. Exiting.{bcolors.ENDC}")
        return list_of_rulesets

    # Doing a 25/75 split of the time budget for generating inputs and refining rules
    time_budget_learner, time_budget_refinement = 0.25*time_budget, 0.75*time_budget
    
    n_variants = get_n_variations(api, lib=lib)
    if suffix > 0 or (suffix == 0 and n_variants == 1):
        variants = [(get_lib_version(api, lib=lib), suffix)]
    else:
        variants = [(get_lib_version(api, lib=lib), i) for i in range(1, n_variants + 1)]

    rng = np.random.default_rng(seed)
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
            print(f"\nStarted invariant inference for {api} (suffix {suff})\n")
            
            ruleset = -1
            valid = 0
            invalid = 0

            start_time = time.time()
            if os.path.isfile(invariant_file):
                print(f"Removing existing invariants file for {variant} at {invariant_file}")
                os.remove(invariant_file)
            
            #### LLM
            if lib == "torch":
                import llm.valid_inputs_torch as valid_inputs
            elif lib == "tf":
                import llm.valid_inputs_tf as valid_inputs
            else:
                raise ValueError(f"Invalid library: {lib}")
            
            api_signature = get_signature(api, lib=lib, suffix=suffix)
            llm_inputs = []
            raw_op_map = get_raw_op_mapping()

            if variant in valid_inputs.generated_inputs:
                print(f"\nAdding LLM generated inputs for {variant}\n")
                llm_inputs = valid_inputs.generated_inputs[variant]
            elif api in raw_op_map:
                print(f"\nUsing inputs from a variant of {api} as {raw_op_map[api]}\n")
                n_variations_new = get_n_variations(raw_op_map[api], lib=lib)
                if n_variations_new == 1:
                    if raw_op_map[api] in valid_inputs.generated_inputs:
                        llm_inputs = valid_inputs.generated_inputs[raw_op_map[api]]
                    else:
                        print(f"{bcolors.WARNING}Warning: {raw_op_map[api]} not found in valid_inputs.generated_inputs{bcolors.ENDC}")
                else:
                    for i in range(1, n_variations_new + 1):
                        variation_new = f"{raw_op_map[api]}_{i}"
                        if variation_new in valid_inputs.generated_inputs:
                            llm_inputs += valid_inputs.generated_inputs[variation_new]
            else:
                print(f"{bcolors.WARNING}Warning: {variant} not found in valid_inputs.generated_inputs{bcolors.ENDC}")

            for llm_input in llm_inputs:
                mutated_inputs = augment_one_input(llm_input, api_signature, lib=lib, rng=rng)
                for mutated_input in mutated_inputs:
                    ruleset, status, exception_message = update_ruleset(api, mutated_input, ruleset=ruleset, lib=lib)
                    if status == "nominal":
                        valid += 1
                    else:
                        invalid += 1
                        if print_details:
                            print(f"[Valid: {valid}, Invalid: {invalid}, Total: {valid + invalid}]")
                            # try:
                            #     print(abstract_print(get_abstract_input(mutated_input, api_signature), api_signature))
                            # except Exception as e:
                            #     print(f"{bcolors.WARNING}Error while printing abstract: {e}{bcolors.ENDC}")
                            print(f"{bcolors.FAIL}Input threw exception: {exception_message}{bcolors.ENDC}")
            ####

            ### Generate and append new inputs (random)
            print(f"\nGenerating inputs for {api} (suffix: {suffix}) with time budget {time_budget_learner} seconds and minimum valid inputs {min_val_inp}\n")
            
            start_time = time.time()
            while (time.time() - start_time < time_budget_learner) and (valid < min_val_inp):                
                input_dict, _ = get_random_input(api_signature, rng, lib=lib)                
                
                mutated_inputs = augment_one_input(input_dict, api_signature, lib=lib, rng=rng)
                for mutated_input in mutated_inputs:
                    ruleset, status, exception_message = update_ruleset(api, mutated_input, ruleset=ruleset, lib=lib)
                    if status == "nominal":
                        valid += 1
                    else:
                        invalid += 1
                        if print_details:
                            print(f"[Valid: {valid}, Invalid: {invalid}, Total: {valid + invalid}]")
                            # try:
                            #     print(abstract_print(get_abstract_input(mutated_input, api_signature), api_signature))
                            # except Exception as e:
                            #     print(f"{bcolors.WARNING}Error while printing abstract: {e}{bcolors.ENDC}")
                            print(f"{bcolors.FAIL}Input threw exception: {exception_message}{bcolors.ENDC}")
            ####
           
            print(f"Invariant inference took {time.time()-start_time:.2f} seconds\n")
            if print_details:
                print_rules(variant, ruleset)

            if len(ruleset) == 0:
                continue
            
            if reduce_rules:
                # Refining stage: If removing a rule does not decrease the validity ratio, remove it
                print(f"Started rule refinement stage for api {api} (suffix {suff}) at {time.asctime()}\n")
                start_time = time.time()
                z3_args = create_z3_args(api_signature)
                ruleset = reduce_ruleset(ruleset, api_signature, api, z3_args, max_trial=min_val_inp, time_budget=time_budget_refinement, print_details=print_details, lib=lib, rng=rng)
                print(f"Rule refinement took {time.time()-start_time:.2f} seconds")
            else:
                print(f"\n{bcolors.WARNING}!!! Skipping rule reduction as per user request. Keeping all {len(ruleset)} rules.{bcolors.ENDC}\n")

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
    # Usage: python -m learner.invariant_inference <variant> <time budget> <1 to regenerate invariants 0 otherwise> <library> <1 to reduce rules 0 otherwise>
    variant = sys.argv[1] if len(sys.argv) > 1 else "scatter"
    budget = int(sys.argv[2]) if len(sys.argv) > 2 else 30  # seconds
    regen = int(sys.argv[3]) == 1 if len(sys.argv) > 3 else False
    lib = sys.argv[4] if len(sys.argv) > 4 else "torch"
    reduce = int(sys.argv[5]) == 1 if len(sys.argv) > 5 else True
    
    api, suffix = get_api_suffix(variant)
    
    ## Since we already have true invariants for some APIs, we skip random generation
    # Try random generation for 60 seconds
    list_of_true_inv_apis = read_file_in_root(f"True_invariants_{lib}")
    invariant_file = os.path.join(get_dir_in_root(f"invariants_{lib}"), variant)
    if variant not in list_of_true_inv_apis:
        # print(f"Running random generation for {api} with suffix {suffix} for 60 seconds to collect baseline validity ratio.")
        # valid, invalid, crash = random_fuzz(api, seed=42, duration=60, lib=lib)
        # if invalid + crash == 0:
        #     print(f"API {api} does not throw exceptions with random inputs after running for 60 seconds. No invariants will be inferred.")
        #     if os.path.isfile(invariant_file):
        #         print(f"Removing existing invariants file for {variant} at {invariant_file}")
        #         os.remove(invariant_file)
        #     append_file_in_root(f"True_invariants_{lib}", f"{variant}\n")
        #     return
        print(f"True invariants for {variant} do not exist. Proceeding with invariant inference.")
    else:
        print(f"True invariants for {variant} already exist. Skipping invariant inference.")
        if os.path.isfile(invariant_file):
            print(f"Removing existing invariants file for {variant} at {invariant_file}")
            os.remove(invariant_file)
        return

    list_of_rulesets = infer_invariants(api, print_details=True, regen=regen, time_budget=budget, z3=True, lib=lib, suffix=suffix, reduce_rules=reduce)

if __name__ == "__main__":
    main()
