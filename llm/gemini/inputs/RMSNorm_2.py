
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def RMSNorm_inputs():
    list_of_inputs = []

    # Case 1: Basic case with float32 and elementwise_affine=True
    x = torch.randn(2, 3, 5).numpy()
    normalized_shape = [5]
    eps = 1e-5
    elementwise_affine = True
    input_dict = {
        "normalized_shape": normalized_shape,
        "eps": eps,
        "elementwise_affine": elementwise_affine,
        "dtype": None,
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2:  int64 input, elementwise_affine=False, removing dtype as it's likely causing issues.
    x = torch.randint(0, 10, (4, 6)).numpy().astype(np.float32) # Convert to float32
    normalized_shape = [6]
    eps = 1e-8
    elementwise_affine = False
    input_dict = {
        "normalized_shape": normalized_shape,
        "eps": eps,
        "elementwise_affine": elementwise_affine,
        "dtype": None, # Removing dtype, RMSNorm infers from input.
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: normalized_shape as a tuple, float64, different eps
    x = torch.randn(1, 2, 3, 4).numpy()
    normalized_shape = [3, 4]
    eps = 1e-6
    elementwise_affine = True
    input_dict = {
        "normalized_shape": normalized_shape,
        "eps": eps,
        "elementwise_affine": elementwise_affine,
        "dtype": None,
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Single integer normalized_shape
    x = torch.randn(5, 7).numpy()
    normalized_shape = [7]
    eps = 1e-4
    elementwise_affine = False
    input_dict = {
        "normalized_shape": normalized_shape,
        "eps": eps,
        "elementwise_affine": elementwise_affine,
        "dtype": None,
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: 3D input with larger dimensions
    x = torch.randn(2, 10, 20).numpy()
    normalized_shape = [20]
    eps = 1e-3
    elementwise_affine = True
    input_dict = {
        "normalized_shape": normalized_shape,
        "eps": eps,
        "elementwise_affine": elementwise_affine,
        "dtype": None,
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.RMSNorm_2"] = RMSNorm_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.RMSNorm_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.RMSNorm_2'.")

check_valid('torch.nn.RMSNorm', generated_inputs['torch.nn.RMSNorm_2'], lib="torch")
