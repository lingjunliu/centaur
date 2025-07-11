import torch
import copy
import time
from utils.new_api_utils import get_signature, get_lib_version, get_n_variations, match_signature_to_input
from utils.misc import get_dir_in_root, save_to_new_pkl, read_pkl, read_file_in_root, bcolors
from generator.input_generators import get_random_input, get_abstract_input, concretize_input
from eval.oracle import oracle_crash
from utils.defaults import domain_limits_torch, domain_limits_tf
import llm.valid_inputs_torch as valid_inputs_torch
import llm.valid_inputs_tf as valid_inputs_tf
import numpy as np
import os
import traceback
import sys

def introduce_float_types(input_dict, signature, lib="torch"):
    """
    Mutation to prevent learning rule_8 incorrectly
    """
    seed = 42
    mutated_inputs = []
    float_types = [np.float16, np.float32, np.float64]
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(float_types))
    for arg, domain in signature.items():
        if input_dict[arg] is None:
            continue
        
        if domain in ["tensor", "tensor_list"]:
            new_input = copy.deepcopy(input_dict)
            if isinstance(new_input[arg], np.ndarray):
                new_input[arg] = new_input[arg].astype(float_types[index%len(float_types)])
            elif isinstance(new_input[arg], list):
                for i, _ in enumerate(new_input[arg]):
                    new_input[arg][i] = new_input[arg][i].astype(float_types[index%len(float_types)])
            else:
                new_input[arg] = float_types[index%len(float_types)](new_input[arg])
            index += 1
            mutated_inputs.append(new_input)
    
    return mutated_inputs

def introduce_floats(input_dict, signature, lib="torch"):
    """
    Mutation to introduce random float values for float-type fields,
    float-typed tensors, or float-valued tuples/lists.
    """
    seed = 42
    mutated_inputs = []
    rng = np.random.default_rng(seed)
    domain_limits = domain_limits_torch if lib == "torch" else domain_limits_tf
    float_min, float_max = domain_limits['float'][:2]

    for arg, domain in signature.items():
        if input_dict[arg] is None:
            continue

        new_input = copy.deepcopy(input_dict)
        if domain == "float":
            new_input[arg] = float(rng.uniform(float_min, float_max))
            mutated_inputs.append(new_input)

        elif domain == "tensor":
            if isinstance(new_input[arg], np.ndarray) and np.issubdtype(new_input[arg].dtype, np.floating):
                new_input[arg] = rng.uniform(float_min, float_max, size=new_input[arg].shape).astype(new_input[arg].dtype)
                mutated_inputs.append(new_input)

        elif domain == "tensor_list":
            if isinstance(new_input[arg], list):
                modified = False
                for i, arr in enumerate(new_input[arg]):
                    if isinstance(arr, np.ndarray) and np.issubdtype(arr.dtype, np.floating):
                        new_input[arg][i] = rng.uniform(float_min, float_max, size=arr.shape).astype(arr.dtype)
                        modified = True
                if modified:
                    mutated_inputs.append(new_input)

        elif domain in ["tuple", "list"]:
            values = new_input[arg]
            if isinstance(values, (tuple, list)):
                modified = False
                new_values = []
                for v in values:
                    if isinstance(v, (float, np.floating)):
                        new_values.append(float(rng.uniform(float_min, float_max)))
                        modified = True
                    else:
                        new_values.append(v)
                if modified:
                    new_input[arg] = tuple(new_values) if domain == "tuple" else new_values
                    mutated_inputs.append(new_input)

    return mutated_inputs

def introduce_integer_types(input_dict, signature, lib="torch"):
    """
    Mutation to prevent learning rule_13 incorrectly
    """
    seed = 42
    mutated_inputs = []
    int_types = [np.int8, np.int16, np.int32, np.int64, np.uint8]
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(int_types))
    for arg, domain in signature.items():
        if input_dict[arg] is None:
            continue
        
        if domain in ["tensor", "tensor_list"]:
            new_input = copy.deepcopy(input_dict)
            if isinstance(new_input[arg], np.ndarray):
                new_input[arg] = new_input[arg].astype(int_types[index%len(int_types)])
            elif isinstance(new_input[arg], list):
                for i, _ in enumerate(new_input[arg]):
                    new_input[arg][i] = new_input[arg][i].astype(int_types[index%len(int_types)])
            else:
                new_input[arg] = int_types[index%len(int_types)](new_input[arg])
            index += 1
            mutated_inputs.append(new_input)
    
    return mutated_inputs

def introduce_integers(input_dict, signature, lib="torch"):
    """
    Mutation to introduce random integer values for int-type fields,
    int-typed tensors, or int-valued tuples/lists.
    """
    seed = 42
    mutated_inputs = []
    rng = np.random.default_rng(seed)
    domain_limits = domain_limits_torch if lib == "torch" else domain_limits_tf
    int_min, int_max = domain_limits['integer'][:2]

    for arg, domain in signature.items():
        if input_dict[arg] is None:
            continue

        new_input = copy.deepcopy(input_dict)
        if domain == "int":
            new_input[arg] = int(rng.integers(int_min, int_max + 1))
            mutated_inputs.append(new_input)

        elif domain == "tensor":
            if isinstance(new_input[arg], np.ndarray) and np.issubdtype(new_input[arg].dtype, np.integer):
                new_input[arg] = rng.integers(int_min, int_max + 1, size=new_input[arg].shape).astype(new_input[arg].dtype)
                mutated_inputs.append(new_input)

        elif domain == "tensor_list":
            if isinstance(new_input[arg], list):
                modified = False
                for i, arr in enumerate(new_input[arg]):
                    if isinstance(arr, np.ndarray) and np.issubdtype(arr.dtype, np.integer):
                        new_input[arg][i] = rng.integers(int_min, int_max + 1, size=arr.shape).astype(arr.dtype)
                        modified = True
                if modified:
                    mutated_inputs.append(new_input)

        elif domain in ["tuple", "list"]:
            values = new_input[arg]
            if isinstance(values, (tuple, list)):
                modified = False
                new_values = []
                for v in values:
                    if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
                        new_values.append(int(rng.integers(int_min, int_max + 1)))
                        modified = True
                    else:
                        new_values.append(v)
                if modified:
                    new_input[arg] = tuple(new_values) if domain == "tuple" else new_values
                    mutated_inputs.append(new_input)

    return mutated_inputs

def introduce_empty_tensors(input_dict, signature, lib="torch"):
    """
    Mutation to prevent learning rule_14 incorrectly
    """
    mutated_inputs = []
    for arg, domain in signature.items():
        if domain in ["tensor", "tensor_list"]:
            new_input = copy.deepcopy(input_dict)
            new_input[arg] = np.array([])
            mutated_inputs.append(new_input)
    
    return mutated_inputs

def introduce_zeros(input_dict, signature, lib="torch"):
    """
    Mutation to prevent learning rule_21 incorrectly
    """
    mutated_inputs = []
    for arg, domain in signature.items():
        new_input = None
        if domain in ["tensor", "tensor_list"]:
            new_input = copy.deepcopy(input_dict)
            if isinstance(new_input[arg], np.ndarray):
                new_input[arg] = np.zeros(new_input[arg].shape)
            else:
                new_input[arg] = 0.0
            
        elif domain == "integer":
            new_input = copy.deepcopy(input_dict)
            new_input[arg] = 0
        elif domain == "float":
            new_input = copy.deepcopy(input_dict)
            new_input[arg] = 0.0
        elif domain in ["list", "tuple"]:
            new_input = copy.deepcopy(input_dict)
            
            if new_input[arg] is None:
                new_input[arg] = [0]
            else:
                if domain == "tuple":
                    new_input[arg] = list(new_input[arg])
                for i, entry in enumerate(new_input[arg]):
                    new_input[arg][i] = 0
                
            if domain == "tuple":
                new_input[arg] = tuple(new_input[arg])
        
        if new_input:
            mutated_inputs.append(new_input)

    return mutated_inputs

def introduce_opposite_bools(input_dict, signature, lib="torch"):
    """
    Mutation to increase diversity
    """
    mutated_inputs = []
    for arg, domain in signature.items():
        if domain == "boolean":
            new_input = copy.deepcopy(input_dict)
            new_input[arg] = not new_input[arg]
            mutated_inputs.append(new_input)
            
    return mutated_inputs

def introduce_negatives(input_dict, signature, lib="torch"):
    """
    Mutation to prevent learning rule_17 and rule_18 incorrectly
    """
    mutated_inputs = []
    none_replacements = {
        "integer": -1,
        "float": -1.0,
        "list": [-1],
        "tuple": (-1),
        "tensor": np.array([-1.0]),
        "tensor_list": np.array([-1.0])
    }
    for arg, domain in signature.items():
        new_input = None
        if input_dict[arg] is None and domain in none_replacements:
            new_input = copy.deepcopy(input_dict)
            new_input[arg] = none_replacements[domain]
        elif domain in ["tensor", "tensor_list", "integer", "float"]:
            new_input = copy.deepcopy(input_dict)
            new_input[arg] = new_input[arg] * -1
        elif domain in ["list", "tuple"]:
            new_input = copy.deepcopy(input_dict)
            for i, entry in enumerate(new_input[arg]):
                if domain == "tuple":
                    new_input[arg] = list(new_input[arg])
            
                for i, entry in enumerate(new_input[arg]):
                    if new_input[arg][i] is None:
                        new_input[arg][i] = -1
                    else:
                        new_input[arg][i] = new_input[arg][i] * -1
                    
                if domain == "tuple":
                    new_input[arg] = tuple(new_input[arg])
        
        if new_input:
            mutated_inputs.append(new_input)

    return mutated_inputs

def augment_inputs(list_of_inputs, signature, lib="torch"):
    """
    Mutate inputs to have diversity to ensure wrong invariants are not learned
    And return the original inputs + mutated inputs
    """
    mutators = [introduce_empty_tensors, introduce_float_types, introduce_floats, introduce_integer_types, introduce_integers, introduce_negatives, introduce_opposite_bools, introduce_zeros]
    original_inputs = []
    mutated_inputs = []
    for i, input_dict in enumerate(list_of_inputs):
        if not match_signature_to_input(input_dict, signature, match_type=True):
            print(f"{bcolors.WARNING}Skipping input at index {i} as it does not match the signature:\n{signature}{bcolors.ENDC}")
            continue  # Skip inputs that do not match the signature
        original_inputs.append(input_dict)
        for mutator in mutators:
            mutated_inputs += mutator(input_dict, signature, lib=lib)
    
    return original_inputs + mutated_inputs

def get_inputs(api, lib="torch", time_budget=30, min_val_inp=100, seed=42, suffix=0):
    """
    Get inputs for an API.
    If LLM generated inputs are available, append them to the list.
    If there are saved inputs, read from that and append to the list.
    At the end, generate new inputs with the random generator using the time budget.
    Augment the inputs with the mutators.
    Return the list of inputs.
    """
    
    list_of_inputs = []
    
    if lib == "torch":
        valid_inputs = valid_inputs_torch
    elif lib == "tf":
        valid_inputs = valid_inputs_tf
    else:
        raise ValueError(f"Invalid library: {lib}")
    
    try:
        api_signature = get_signature(api, lib=lib, suffix=suffix)
    except Exception as e:
        # Could not get signature, return empty list
        print(f"{bcolors.FAIL}Error getting signature for {api} | {e.__class__.__name__}: {e}{bcolors.ENDC}")
        return []
    
    lib_api = get_lib_version(api, lib=lib)
    variation = f"{lib_api}_{suffix}" if suffix > 0 else lib_api
    # Return LLM generated inputs if available
    if variation in valid_inputs.generated_inputs:
        print(f"Adding LLM generated inputs for {variation}")
        list_of_inputs = valid_inputs.generated_inputs[variation]

    input_file = os.path.join(get_dir_in_root(f"valid_inputs_{lib}"), f"{api}.pkl")
    
    # If there already is a saved file, read from that and concretize
    if os.path.isfile(input_file):
        print(f"Adding saved inputs for {api} from {input_file}")
        abstract_inputs = read_pkl(input_file)
        for abs_inp, saved_seed, suff in abstract_inputs:
            if suff != suffix:
                continue
            rng = np.random.default_rng(saved_seed)
            list_of_inputs.append(concretize_input(abs_inp, api_signature, rng))
    
    # Generate and append new inputs
    print(f"Generating inputs for {api} (suffix: {suffix}) with time budget {time_budget} seconds and minimum valid inputs {min_val_inp}")
    valid = 0
    abstract_inputs = []
    
    start_time = time.time()
    while (time.time() - start_time < time_budget) and (valid < min_val_inp):
        rng = np.random.default_rng(seed)
        input_dict, abs_inp = get_random_input(api_signature, rng, lib=lib)
        status, exception_message = oracle_crash(api, input_dict, cpu=True, lib=lib)
        if status == "nominal":
            valid += 1
            list_of_inputs.append(input_dict)
            abstract_inputs.append((abs_inp, seed, suffix))
        
        seed += 1
    
    # Save abstract inputs to file
    save_to_new_pkl(input_file, abstract_inputs)
    
    list_of_inputs = augment_inputs(list_of_inputs, api_signature, lib=lib)
    
    return list_of_inputs

def main():
    all_apis = set(read_file_in_root("torch_apis.txt"))
    apis = set()
    total_inputs = 0
    apis_with_issues = set()
    
    lib = sys.argv[1] if len(sys.argv) > 1 else "torch"
    if lib == "torch":
        valid_inputs = valid_inputs_torch
    elif lib == "tf":
        valid_inputs = valid_inputs_tf
    else:
        raise ValueError(f"Invalid library: {lib}")
    
    for api in all_apis:
        n_variations = get_n_variations(api, lib=lib)
        suffixes = []
        if n_variations == 1:
            suffixes = [0]
        else:
            suffixes = [i for i in range(1, n_variations + 1)]

        for suffix in suffixes:
            variation = f"{api}_{suffix}" if suffix > 0 else api
            if variation not in valid_inputs.generated_inputs.keys():
                print(f"{bcolors.WARNING}Warning: {variation} not found in valid_inputs.generated_inputs{bcolors.ENDC}")
                apis_with_issues.add(api)
                break
            generated_inputs = get_inputs(api, time_budget=0, lib=lib, suffix=suffix)
            if len(generated_inputs) == 0:
                print(f"{bcolors.WARNING}Warning: No inputs generated for {api} with suffix {suffix}{bcolors.ENDC}")
                apis_with_issues.add(api)
                break
            apis.add(api)
            total_inputs += len(generated_inputs)

            for input_dict in generated_inputs:
                try:
                    _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
                except Exception as e:
                    print(f"{bcolors.FAIL}Error getting abstract input for {api} with suffix {suffix} | {e.__class__.__name__}: {e}{bcolors.ENDC}")
                    apis_with_issues.add(api)
                    break

    
    print(f"\n{len(apis)} apis has pre-defined inputs, {round(total_inputs/len(apis), 2) if len(apis) > 0 else 0} inputs on average")
    print(f"{len(apis_with_issues)} APIs with issues, {len(all_apis - apis)} APIs without pre-defined inputs, {len(all_apis)} APIs in total")
    
    predefined_inputs_file = os.path.join(get_dir_in_root("llm"), "predefined_inputs.txt")
    needs_inputs_file = os.path.join(get_dir_in_root("llm"), "needs_inputs.txt")
    issues_file = os.path.join(get_dir_in_root("llm"), "apis_w_problematic_inputs.txt")
    
    with open(predefined_inputs_file, "w") as f:
        for api in sorted(apis):
            f.write(f"{api}\n")
    with open(needs_inputs_file, "w") as f:
        for api in sorted(all_apis - apis):
            f.write(f"{api}\n")
    with open(issues_file, "w") as f:
        for api in sorted(apis_with_issues):
            f.write(f"{api}\n")
    
if __name__ == "__main__":
    main()
