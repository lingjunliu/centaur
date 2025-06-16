import torch
import copy
import time
from utils.api_utils import get_signatures, get_driver
from utils.misc import get_dir_in_root, map_torch_to_driver, save_to_new_pkl, read_pkl, read_file_in_root
from generator.input_generators import get_random_input, get_abstract_input, concretize_input
from eval.oracle import oracle_crash
import llm.valid_inputs as valid_inputs
import numpy as np
import os

def scatter_inputs():
    list_of_inputs = []
    # Input 1, valid
    src_torch = torch.arange(1, 11).reshape((2, 5))
    src = src_torch.numpy() 
    index = torch.tensor([[0, 1, 2, 0]]).numpy()
    input = torch.zeros(3, 5, dtype=src_torch.dtype).numpy()
    dim = 0

    input_dict = {
        "input": input,
        "dim": dim,
        "src": src, 
        "index": index,         
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, invalid
    index_2 = torch.tensor([[0, 1, 2], [0, 1, 4]])
    input_dict["index"] = index_2.numpy()
    
    # Skipping invalid inputs
    # list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid
    input_dict["dim"] = 1
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, invalid
    input_dict = {
        "input": torch.full((2, 4), 2.).numpy(),
        "dim": 1,
        "src": 1.23, 
        "index": torch.tensor([[2], [3]]).numpy(),         
    }
    
    # Skipping invalid inputs
    # list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid
    input_dict = {
        "input": torch.full((2, 4), 2., dtype=int).numpy(),
        "dim": 1,
        "src": torch.tensor([[2], [3]]).numpy(), 
        "index": torch.tensor([[2], [3]]).numpy(),         
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input_dict = {
        "input": torch.full((2, 4), 2., dtype=torch.float32).numpy(),
        "dim": 1,
        "src": torch.tensor([[2], [3]], dtype=torch.float32).numpy(), 
        "index": torch.tensor([[2], [3]]).numpy(),         
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

def matmul_inputs():
    list_of_inputs = []
    # Input 1, valid
    input = torch.randn(3, 5).numpy()
    other = torch.randn(5, 2).numpy() 

    input_dict = {
        "input": input,
        "other": other
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.randn(3, 5).numpy()
    other = torch.randn(5).numpy()

    input_dict = {
        "input": input,
        "other": other
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

def combinations_inputs():
    list_of_inputs = []
    # Input 1, valid
    input = torch.tensor([10, 20, 30, 40]).numpy()
    r = 3
    with_replacement = False

    input_dict = {
        "input": input,
        "r": r,
        "with_replacement": with_replacement
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([1.5, 2.5]).numpy()
    r = 2
    with_replacement = True

    input_dict = {
        "input": input,
        "r": r,
        "with_replacement": with_replacement
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

def addcmul_inputs():
    list_of_inputs = []
    # Input 1, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    tensor1 = torch.tensor([0.1, 0.2, 0.3]).numpy()
    tensor2 = torch.tensor([10.0, 20.0, 30.0]).numpy()
    value = 2.0  

    input_dict = {
        "input": input,
        "tensor1": tensor1,
        "tensor2": tensor2,
        "value": value
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.ones((2, 3)).numpy()
    tensor1 = torch.tensor([[1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0]]).numpy()
    tensor2 = torch.tensor([[0.1, 0.2, 0.3],
                            [0.4, 0.5, 0.6]]).numpy()
    value = 0.5

    input_dict = {
        "input": input,
        "tensor1": tensor1,
        "tensor2": tensor2,
        "value": value
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

def lp_pool1d_inputs():
    list_of_inputs = []
    # Input 1, valid
    input = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0]]]).numpy()
    norm_type = 2.0
    kernel_size = 2
    stride = 2
    ceil_mode = True
    
    input_dict = {
        "input": input,
        "norm_type": norm_type,
        "kernel_size": kernel_size,
        "stride": stride,
        "ceil_mode": ceil_mode
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid
    input = torch.tensor([[[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]]]).numpy()
    norm_type = 1.0
    kernel_size = 3
    stride = 2
    ceil_mode = True
    
    input_dict = {
        "input": input,
        "norm_type": norm_type,
        "kernel_size": kernel_size,
        "stride": stride,
        "ceil_mode": ceil_mode
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

def introduce_floats(input_dict, signature):
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

def introduce_integers(input_dict, signature):
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

def introduce_empty_tensors(input_dict, signature):
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

def introduce_zeros(input_dict, signature):
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

def introduce_opposite_bools(input_dict, signature):
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

def introduce_negatives(input_dict, signature):
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

def augment_inputs(list_of_inputs, signature):
    """
    Mutate inputs to have diversity to ensure wrong invariants are not learned
    And return the original inputs + mutated inputs
    """
    mutators = [introduce_empty_tensors, introduce_floats, introduce_integers, introduce_negatives, introduce_opposite_bools, introduce_zeros]
    mutated_inputs = []
    for input_dict in list_of_inputs:
        for arg,domain in signature.items():
            if arg not in input_dict:
                print(f"\nSignature: {signature} | input: {input_dict.keys()}\n")
        for mutator in mutators:
            mutated_inputs += mutator(input_dict, signature)
            
    return list_of_inputs + mutated_inputs

# Add human and LLM defined inputs for APIs that are
# difficult to generate inputs for
inputs_per_api = {
    "scatter": scatter_inputs(),      # human start
    "matmul": matmul_inputs(),
    "combinations": combinations_inputs(),
    "addcmul": addcmul_inputs(),
    "lp_pool1d_": lp_pool1d_inputs(), # human end
}

def get_inputs(api, lib="torch", time_budget=30, min_val_inp=5, seed=42):
    api_signature = get_signatures()[api]
    torch_to_driver, driver_to_torch = map_torch_to_driver()
    
    # Return LLM generated inputs if available
    if driver_to_torch[api] in valid_inputs.generated_inputs:
        try:
            return augment_inputs(valid_inputs.generated_inputs[driver_to_torch[api]], api_signature)
        except Exception as e:
            print(f"Error augmenting inputs for {driver_to_torch[api]} | {e.__class__.__name__}: {e}")

    # Return human written inputs if available
    if api in inputs_per_api:
        try:
            return augment_inputs(inputs_per_api[api], api_signature)
        except Exception as e:
            print(f"Error augmenting inputs for {api} in {lib} | {e.__class__.__name__}: {e}")
    
    # Generate valid inputs through random generation otherwise
    input_file = os.path.join(get_dir_in_root(f"valid_inputs_{lib}"), f"{api}.pkl")
    
    list_of_inputs = []
    
    # If there already is a saved file, read from that and concretize
    if os.path.isfile(input_file):
        abstract_inputs = read_pkl(input_file)
        for abs_inp, saved_seed in abstract_inputs:
            rng = np.random.default_rng(saved_seed)
            list_of_inputs.append(concretize_input(abs_inp, api_signature, rng))
    else:   # Generate and save otherwise
        api_driver = get_driver(api, lib=lib)
        valid = 0
        invalid = 0
        abstract_inputs = []
        
        start_time = time.time()
        while (time.time() - start_time < time_budget) and (valid < min_val_inp):
            rng = np.random.default_rng(seed)
            input_dict = get_random_input(api_signature, rng)
            status, exception_message = oracle_crash(api, api_signature, input_dict, cpu=True, lib=lib)
            if status == "invalid":
                invalid += 1
            else:
                valid += 1
                # Only adding valid inputs
                list_of_inputs.append(input_dict)
                abs_inp = get_abstract_input(input_dict, api_signature)
                # Save the abstract input along with the seed
                abstract_inputs.append((abs_inp, seed))
            
            seed += 1
        
        # Save abstract inputs to file
        save_to_new_pkl(input_file, abstract_inputs)
    
    return augment_inputs(list_of_inputs, api_signature)

def main():
    all_apis = set(read_file_in_root("apis.txt"))
    apis = set()
    total_inputs = 0
    torch_to_driver, driver_to_torch = map_torch_to_driver()
    apis_with_issues = set()
    for api, inputs in inputs_per_api.items():
        generated_inputs = get_inputs(api)
        if len(generated_inputs) == 0:
            print(f"Warning: No inputs generated for {api}")
            apis_with_issues.add(api)
            continue
        apis.add(api)
        total_inputs += len(generated_inputs)
    for torch_api, inputs in valid_inputs.generated_inputs.items():
        api = torch_to_driver[torch_api]
        generated_inputs = get_inputs(api)
        if len(generated_inputs) == 0:
            print(f"Warning: No inputs generated for {api}")
            apis_with_issues.add(api)
            continue
        apis.add(api)
        total_inputs += len(generated_inputs)
    
    print(f"\n{len(apis)} apis has pre-defined inputs, {round(total_inputs/len(apis), 2)} inputs on average")
    print(f"{len(apis_with_issues)} APIs with issues, {len(all_apis - apis)} APIs without pre-defined inputs, {len(all_apis)} APIs in total")
    
    predefined_inputs_file = os.path.join(get_dir_in_root("llm"), "predefined_inputs.txt")
    needs_inputs_file = os.path.join(get_dir_in_root("llm"), "needs_inputs.txt")
    
    with open(predefined_inputs_file, "w") as f:
        for api in sorted(apis):
            f.write(f"{api}\n")
    with open(needs_inputs_file, "w") as f:
        for api in sorted(all_apis - apis):
            f.write(f"{api}\n")
    
    print("APIs with issues:")
    for api in sorted(apis_with_issues):
        print(f"{api}")
    
if __name__ == "__main__":
    main()