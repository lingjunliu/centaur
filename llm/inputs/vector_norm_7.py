
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def vector_norm_inputs():
    list_of_inputs = []

    input = np.array([1, 2, 3], dtype=np.float32)
    input_dict = {
        "input": input,
        "ord": 2.0,
        "dim": None,
        "keepdim": False,
        "dtype": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = np.array([[1, 2], [3, 4]], dtype=np.float32)
    input_dict = {
        "input": input,
        "ord": 2.0,
        "dim": None,
        "keepdim": False,
        "dtype": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = np.array([[-1, 2], [3, -4]], dtype=np.float32)
    input_dict = {
        "input": input,
        "ord": 1.0,
        "dim": 0,
        "keepdim": True,
        "dtype": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = np.array([1, 2, 3], dtype=np.float64)
    input_dict = {
        "input": input,
        "ord": np.inf,
        "dim": None,
        "keepdim": False,
        "dtype": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = np.array([[1, 2], [3, 4]], dtype=np.float32)
    input_dict = {
        "input": input,
        "ord": -1.0,
        "dim": 0,
        "keepdim": True,
        "dtype": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.linalg.vector_norm_7"] = vector_norm_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.linalg.vector_norm_7' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.vector_norm_7'.")

check_valid('torch.linalg.vector_norm', generated_inputs['torch.linalg.vector_norm_7'], lib="torch")
