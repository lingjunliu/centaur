
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def vector_norm_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor, L2 norm
    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict1 = {
        "input": input1,
        "ord": 2.0,
        "dim": None,
        "keepdim": False,
        "dtype": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Integer tensor, L1 norm, specifying dtype
    input2 = np.array([-1.0, 2.0, -3.0], dtype=np.float32) # Changed to float32
    input_dict2 = {
        "input": input2,
        "ord": 1.0,
        "dim": None,
        "keepdim": False,
        "dtype": torch.float64,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 2D float tensor, Frobenius norm (L2 norm for matrices)
    input3 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict3 = {
        "input": input3,
        "ord": 2.0, # Changed from 'fro' to 2.0 for Frobenius
        "dim": None,
        "keepdim": False,
        "dtype": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 3D float tensor, L2 norm along dimension 1, keep dimension
    input4 = np.random.rand(2, 3, 4).astype(np.float32)
    input_dict4 = {
        "input": input4,
        "ord": 2.0,
        "dim": 1,
        "keepdim": True,
        "dtype": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5: Float tensor, inf norm
    input5 = np.array([-1.0, 2.0, -3.0, 4.0], dtype=np.float32)
    input_dict5 = {
        "input": input5,
        "ord": np.inf,
        "dim": None,
        "keepdim": False,
        "dtype": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs

generated_inputs["torch.linalg.vector_norm_2"] = vector_norm_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.linalg.vector_norm_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.vector_norm_2'.")

check_valid('torch.linalg.vector_norm', generated_inputs['torch.linalg.vector_norm_2'], lib="torch")
