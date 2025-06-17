
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def addr_inputs():
    list_of_inputs = []

    # Test case 1: Basic float tensors
    input1 = np.zeros((3, 2), dtype=np.float32)
    vec1_1 = np.arange(1, 4, dtype=np.float32)
    vec2_1 = np.arange(1, 3, dtype=np.float32)
    input_dict1 = {
        "input": input1,
        "vec1": vec1_1,
        "vec2": vec2_1,
        "beta": 1.0,
        "alpha": 1.0,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Test case 2: Different beta and alpha values
    input2 = np.ones((4, 2), dtype=np.float64)
    vec1_2 = np.array([-1, 0, 1, 2], dtype=np.float64)
    vec2_2 = np.array([2, -1], dtype=np.float64)
    input_dict2 = {
        "input": input2,
        "vec1": vec1_2,
        "vec2": vec2_2,
        "beta": 0.5,
        "alpha": 2.0,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Test case 3: Integer tensors
    input3 = np.zeros((4, 3), dtype=np.int64)
    vec1_3 = np.array([1, 2, 3, 4], dtype=np.int64)
    vec2_3 = np.array([5, 6, 7], dtype=np.int64)
    input_dict3 = {
        "input": input3,
        "vec1": vec1_3,
        "vec2": vec2_3,
        "beta": 1,
        "alpha": 1,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Test case 4: beta = 0
    input4 = np.full((2, 2), 5.0, dtype=np.float32)
    vec1_4 = np.array([1.0, 2.0], dtype=np.float32)
    vec2_4 = np.array([3.0, 4.0], dtype=np.float32)
    input_dict4 = {
        "input": input4,
        "vec1": vec1_4,
        "vec2": vec2_4,
        "beta": 0.0,
        "alpha": 1.0,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Test case 5: Providing an output tensor
    input5 = np.zeros((5, 4), dtype=np.float32)
    vec1_5 = np.arange(1, 6, dtype=np.float32)
    vec2_5 = np.arange(1, 5, dtype=np.float32)
    out_5 = np.empty((5, 4), dtype=np.float32)
    input_dict5 = {
        "input": input5,
        "vec1": vec1_5,
        "vec2": vec2_5,
        "beta": 1.0,
        "alpha": 1.0,
        "out": out_5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.addr"] = addr_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.addr' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.addr'.")

check_valid('torch.addr', generated_inputs['torch.addr'], lib="torch")
