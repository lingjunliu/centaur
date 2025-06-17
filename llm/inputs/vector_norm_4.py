
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def vector_norm_inputs():
    list_of_inputs = []

    input_1 = np.array([1, 2, 3, 4], dtype=np.float32)
    input_dict_1 = {
        "input": input_1,
        "ord": 2,
        "dim": None,
        "keepdim": False,
        "dtype": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    input_2 = np.array([[1, 2], [3, 4]], dtype=np.float32)
    input_dict_2 = {
        "input": input_2,
        "ord": 1,
        "dim": [0],
        "keepdim": True,
        "dtype": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    input_3 = np.array([[-1, 2], [-3, 4]], dtype=np.float64)
    input_dict_3 = {
        "input": input_3,
        "ord": np.inf,
        "dim": [1],
        "keepdim": False,
        "dtype": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    input_4 = np.array([1, 2, 3, 4], dtype=np.float64)
    input_dict_4 = {
        "input": input_4,
        "ord": -np.inf,
        "dim": None,
        "keepdim": False,
        "dtype": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    input_5 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    input_dict_5 = {
        "input": input_5,
        "ord": 2,
        "dim": [1, 2],
        "keepdim": True,
        "dtype": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    input_6 = np.array([1, 2, 3, 4], dtype=np.float32)
    input_dict_6 = {
        "input": input_6,
        "ord": -1,
        "dim": None,
        "keepdim": False,
        "dtype": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    input_7 = np.array([[1, 2], [3, 4]], dtype=np.float32)
    input_dict_7 = {
        "input": input_7,
        "ord": 2,
        "dim": [0, 1],
        "keepdim": False,
        "dtype": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    return list_of_inputs

generated_inputs["torch.linalg.vector_norm_4"] = vector_norm_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.linalg.vector_norm_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.vector_norm_4'.")

check_valid('torch.linalg.vector_norm', generated_inputs['torch.linalg.vector_norm_4'], lib="torch")
