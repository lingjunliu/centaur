
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def vector_norm_inputs():
    list_of_inputs = []

    # Case 1: Basic case with float tensor and default parameters
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

    # Case 2: Float tensor with ord=1 (L1 norm)
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

    # Case 3: 2D float tensor, dim=1, keepdim=True. Remove fro for now.
    input_tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {
        "input": input_tensor,
        "ord": 2,
        "dim": [1],
        "keepdim": True,
        "dtype": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Complex tensor with ord=2 (L2 norm)
    input_tensor = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex64)
    input_dict = {
        "input": input_tensor,
        "ord": 2,
        "dim": None,
        "keepdim": False,
        "dtype": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 5: 3D float tensor, dim=(0, 1). Remove fro for now.
    input_tensor = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "ord": 2,
        "dim": [0, 1],
        "keepdim": False,
        "dtype": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: inf norm
    input_tensor = np.array([-1.0, 2.0, -3.0], dtype=np.float32)
    input_dict = {
        "input": input_tensor,
        "ord": np.inf,
        "dim": None,
        "keepdim": False,
        "dtype": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7: -inf norm
    input_tensor = np.array([-1.0, 2.0, -3.0], dtype=np.float32)
    input_dict = {
        "input": input_tensor,
        "ord": -np.inf,
        "dim": None,
        "keepdim": False,
        "dtype": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.linalg.vector_norm_8"] = vector_norm_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.linalg.vector_norm_8' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.vector_norm_8'.")

check_valid('torch.linalg.vector_norm', generated_inputs['torch.linalg.vector_norm_8'], lib="torch")
