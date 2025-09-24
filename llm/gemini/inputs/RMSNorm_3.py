
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy
import numpy as np

def rmsnorm_inputs():
    list_of_inputs = []

    # Case 1: Basic case with normalized_shape as a tuple, float32
    input_tensor = torch.randn(2, 3, 4).numpy()
    input_dict = {
        "normalized_shape": (4,),
        "eps": 1e-08,
        "elementwise_affine": True,
        "dtype": torch.float32,
        "x": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: normalized_shape as an int, float64
    input_tensor = torch.randn(2, 5).numpy()
    input_dict = {
        "normalized_shape": (5,),
        "eps": 1e-06,
        "elementwise_affine": False,
        "dtype": torch.float64,
        "x": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Different input dimensions, complex64
    input_tensor = torch.randn(1, 2, 3, 4, dtype=torch.complex64).numpy()
    input_dict = {
        "normalized_shape": (4,),
        "eps": 1e-05,
        "elementwise_affine": True,
        "dtype": torch.complex64,
        "x": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Smaller eps, elementwise_affine=False, float16
    input_tensor = torch.randn(4, 6, dtype=torch.float16).numpy()
    input_dict = {
        "normalized_shape": (6,),
        "eps": 1e-12,
        "elementwise_affine": False,
        "dtype": torch.float16,
        "x": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: normalized shape with multiple dimensions
    input_tensor = torch.randn(2, 3, 5, 7).numpy()
    input_dict = {
        "normalized_shape": (5,7),
        "eps": 1e-08,
        "elementwise_affine": True,
        "dtype": torch.float32,
        "x": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.RMSNorm_3"] = rmsnorm_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.RMSNorm_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.RMSNorm_3'.")

check_valid('torch.nn.RMSNorm', generated_inputs['torch.nn.RMSNorm_3'], lib="torch")
