
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def vector_norm_inputs():
    list_of_inputs = []

    # Input 1: Basic case with L2 norm
    input1 = np.array([1.0, 2.0, 3.0])
    input_dict1 = {
        "input": input1,
        "ord": 2,
        "dim": None,
        "keepdim": False,
        "dtype": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Different order norm (L1)
    input2 = np.array([-1.0, 2.0, -3.0])
    input_dict2 = {
        "input": input2,
        "ord": 1,
        "dim": None,
        "keepdim": False,
        "dtype": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Multidimensional array, norm along a specific dimension
    input3 = np.array([[1.0, 2.0], [3.0, 4.0]])
    input_dict3 = {
        "input": input3,
        "ord": 2,
        "dim": (0,),
        "keepdim": True,
        "dtype": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Integer tensor
    input4 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict4 = {
        "input": input4,
        "ord": 2,
        "dim": None,
        "keepdim": False,
        "dtype": torch.float64,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Complex tensor
    input5 = np.array([1 + 1j, 2 - 2j, 3 + 0j])
    input_dict5 = {
        "input": input5,
        "ord": 2,
        "dim": None,
        "keepdim": False,
        "dtype": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs

generated_inputs["torch.linalg.vector_norm_9"] = vector_norm_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.linalg.vector_norm_9' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.vector_norm_9'.")

check_valid('torch.linalg.vector_norm', generated_inputs['torch.linalg.vector_norm_9'], lib="torch")
