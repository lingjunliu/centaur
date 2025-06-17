
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def var_mean_inputs():
    list_of_inputs = []

    # Input 1: 1D float tensor
    input_1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    input_dict_1 = {
        "input": input_1,
        "unbiased": True,
        "dim": None,
        "keepdim": False,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 2D int tensor with dim specified
    input_2 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    input_dict_2 = {
        "input": input_2,
        "unbiased": False,
        "dim": (0,),
        "keepdim": True,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 3D float tensor with multiple dims
    input_3 = np.random.rand(2, 3, 4).astype(np.float64)
    input_dict_3 = {
        "input": input_3,
        "unbiased": True,
        "dim": (0, 1),
        "keepdim": False,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: 1D tensor with negative values
    input_4 = np.array([-1.0, -2.0, 3.0, 4.0], dtype=np.float32)
    input_dict_4 = {
        "input": input_4,
        "unbiased": False,
        "dim": None,
        "keepdim": False,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))
    
    # Input 5: 2D float tensor along all dimensions
    input_5 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict_5 = {
        "input": input_5,
        "unbiased": True,
        "dim": (0, 1),
        "keepdim": True,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    return list_of_inputs

generated_inputs["torch.var_mean_4"] = var_mean_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.var_mean_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.var_mean_4'.")

check_valid('torch.var_mean', generated_inputs['torch.var_mean_4'], lib="torch")
