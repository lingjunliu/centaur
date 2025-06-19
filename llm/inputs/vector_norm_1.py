
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def vector_norm_inputs():
    list_of_inputs = []

    # Input 1: Basic case with l2 norm and float32
    input_tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {
        "input": input_tensor,
        "ord": 2,
        "dim": None,
        "keepdim": False,
        "dtype": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: l1 norm, float input
    input_tensor = np.array([-1.0, 2.0, -3.0, 4.0], dtype=np.float32)
    input_dict = {
        "input": input_tensor,
        "ord": 1,
        "dim": None,
        "keepdim": False,
        "dtype": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: inf norm, float64 input
    input_tensor = np.array([-1.5, 2.5, -3.5, 4.5], dtype=np.float64)
    input_dict = {
        "input": input_tensor,
        "ord": np.inf,
        "dim": None,
        "keepdim": False,
        "dtype": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: -inf norm
    input_tensor = np.array([-1.5, 2.5, -3.5, 4.5], dtype=np.float64)
    input_dict = {
        "input": input_tensor,
        "ord": -np.inf,
        "dim": None,
        "keepdim": False,
        "dtype": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Matrix with dimension and keepdim
    input_tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {
        "input": input_tensor,
        "ord": 2,
        "dim": 1,
        "keepdim": True,
        "dtype": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 3D tensor
    input_tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    input_dict = {
        "input": input_tensor,
        "ord": 2,
        "dim": (0, 1),
        "keepdim": True,
        "dtype": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Negative ord
    input_tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {
        "input": input_tensor,
        "ord": -2,
        "dim": None,
        "keepdim": False,
        "dtype": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Specify dtype
    input_tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {
        "input": input_tensor,
        "ord": 2,
        "dim": None,
        "keepdim": False,
        "dtype": torch.float64,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.linalg.vector_norm_1"] = vector_norm_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.linalg.vector_norm_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.vector_norm_1'.")

check_valid('torch.linalg.vector_norm', generated_inputs['torch.linalg.vector_norm_1'], lib="torch")
