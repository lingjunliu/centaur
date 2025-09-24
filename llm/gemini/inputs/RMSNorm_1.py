
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy
import numpy as np

def RMSNorm_inputs():
    list_of_inputs = []

    input_tensor = torch.randn(2, 3, 4).numpy()
    normalized_shape = 4
    eps = 1e-05
    elementwise_affine = True
    dtype = torch.float32

    input_dict = {
        "normalized_shape": normalized_shape,
        "eps": eps,
        "elementwise_affine": elementwise_affine,
        "dtype": dtype,
        "x": input_tensor
    }

    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(2, 5).numpy()
    normalized_shape = [5]
    eps = 1e-08
    elementwise_affine = False
    dtype = torch.float64

    input_dict = {
        "normalized_shape": normalized_shape,
        "eps": eps,
        "elementwise_affine": elementwise_affine,
        "dtype": dtype,
        "x": input_tensor
    }

    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(1, 2, 3, 5).numpy()
    normalized_shape = (3, 5)
    eps = 1e-12
    elementwise_affine = True
    dtype = torch.float16

    input_dict = {
        "normalized_shape": normalized_shape,
        "eps": eps,
        "elementwise_affine": elementwise_affine,
        "dtype": dtype,
        "x": input_tensor
    }

    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(4, 2).numpy()
    normalized_shape = 2
    eps = 1e-06
    elementwise_affine = False
    dtype = torch.bfloat16

    input_dict = {
        "normalized_shape": normalized_shape,
        "eps": eps,
        "elementwise_affine": elementwise_affine,
        "dtype": dtype,
        "x": input_tensor
    }

    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(3, 4, 5, 6).numpy()
    normalized_shape = [4,5,6]
    eps = 1e-7
    elementwise_affine = True
    dtype = torch.float32
    input_dict = {
        "normalized_shape": normalized_shape,
        "eps": eps,
        "elementwise_affine": elementwise_affine,
        "dtype": dtype,
        "x": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.RMSNorm_1"] = RMSNorm_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.RMSNorm_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.RMSNorm_1'.")

check_valid('torch.nn.RMSNorm', generated_inputs['torch.nn.RMSNorm_1'], lib="torch")
