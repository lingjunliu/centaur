import torch
import copy
import time
from utils.api_utils import get_signatures, get_driver
from utils.misc import get_dir_in_root
from generator.input_generators import get_random_input, get_abstract_input, concretize_input
import numpy as np
import pickle, os

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

def conv_transpose2d_inputs():
    list_of_inputs = []
    # Input 1, valid
    input = torch.randn(1, 3, 5, 5).numpy()        # [N, C_in, H_in, W_in]
    weight = torch.randn(3, 2, 3, 3).numpy()       # [C_in, C_out, kH, kW]
    stride = 1
    padding = 0
    
    input_dict = {
        "input": input,
        "weight": weight,
        "stride": stride,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid
    input = torch.randn(1, 4, 4, 4).numpy()        # [N, C_in, H_in, W_in]
    weight = torch.randn(4, 2, 3, 3).numpy()       # [C_in, C_out, kH, kW]
    stride = 2
    padding = 1
    
    input_dict = {
        "input": input,
        "weight": weight,
        "stride": stride,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid
    input = torch.randn(1, 2, 6, 6).numpy()        # [N, C_in, H_in, W_in]
    weight = torch.randn(2, 1, 5, 5).numpy()       # [C_in, C_out, kH, kW]
    stride = 1
    padding = 0
    
    input_dict = {
        "input": input,
        "weight": weight,
        "stride": stride,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid
    input = torch.randn(1, 1, 5, 5).numpy()        # [N, C_in, H_in, W_in]
    weight = torch.randn(1, 1, 3, 3).numpy()       # [C_in, C_out, kH, kW]
    stride = 2
    padding = 1
    
    input_dict = {
        "input": input,
        "weight": weight,
        "stride": stride,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid
    input = torch.randn(2, 8, 10, 10).numpy()        # [N, C_in, H_in, W_in]
    weight = torch.randn(8, 16, 4, 4).numpy()       # [C_in, C_out, kH, kW]
    stride = 2
    padding = 1
    
    input_dict = {
        "input": input,
        "weight": weight,
        "stride": stride,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

# Add human defined inputs for APIs that are
# difficult to generate inputs for
inputs_per_api = {
    "scatter": scatter_inputs,
    "conv_transpose2d": conv_transpose2d_inputs
}

def get_inputs(api, lib="torch", time_budget=30, min_val_inp=5, seed=42):
    # Return human written inputs if available
    if api in inputs_per_api:
        return inputs_per_api[api]()
    
    # Generate valid inputs through random generation otherwise
    api_signature = get_signatures()[api]
    input_file = os.path.join(get_dir_in_root("valid_inputs"), f"{api}.pkl")
    
    list_of_inputs = []
    
    # If there already is a saved file, read from that and concretize
    if os.path.isfile(input_file):
        with open(input_file, "rb") as f:
            abstract_inputs = pickle.load(f)
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
            try:
                out_cpu = api_driver(input_dict, cpu=True)
            except:
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
        with open(input_file, "wb") as f:
            pickle.dump(abstract_inputs, f)
            
    return list_of_inputs