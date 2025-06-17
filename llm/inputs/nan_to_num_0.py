
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def nan_to_num_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor with NaN, inf, -inf
    input1 = np.array([float('nan'), float('inf'), float('-inf'), 1.0, 2.0], dtype=np.float32)
    input_dict1 = {
        "input": input1,
        "nan": 0.0,
        "posinf": 1.0,
        "neginf": -1.0,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 3: 2D float tensor
    input3 = np.array([[float('nan'), 1.0], [float('inf'), 2.0]], dtype=np.float64)
    input_dict3 = {
        "input": input3,
        "nan": 0.0,
        "posinf": 1e10,
        "neginf": -1e10,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 3D float tensor with different replacement values
    input4 = np.array([[[float('nan'), 1.0], [float('inf'), 2.0]], [[-3.0, float('-inf')], [4.0, 5.0]]], dtype=np.float32)
    input_dict4 = {
        "input": input4,
        "nan": -999.0,
        "posinf": 999.0,
        "neginf": -999.0,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Tensor with all values as NaN, inf, -inf
    input5 = np.array([float('nan'), float('inf'), float('-inf')], dtype=np.float16)
    input_dict5 = {
        "input": input5,
        "nan": 0.0,
        "posinf": 1e5,
        "neginf": -1e5,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Using numpy array of complex numbers
    input6 = np.array([1 + 1j * float('nan'), float('inf') + 2j, 3 - 1j * float('-inf')], dtype=np.complex64)
    input_dict6 = {
        "input": input6,
        "nan": 0.0,
        "posinf": 1e3,
        "neginf": -1e3,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    return list_of_inputs

generated_inputs["torch.nan_to_num"] = nan_to_num_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nan_to_num' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nan_to_num'.")

check_valid('torch.nan_to_num', generated_inputs['torch.nan_to_num'], lib="torch")
