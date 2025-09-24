
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def GaussianNLLLoss_inputs():
    list_of_inputs = []

    # Case 1: Basic case with different shapes and negative values
    input_np = np.random.randn(5, 2).astype(np.float32)
    target_np = np.random.randn(5, 2).astype(np.float32)
    var_np = np.random.rand(5, 2).astype(np.float32) + 1e-6
    input_dict = {
        "full": False,
        "eps": 1e-6,
        "reduction": 'mean',
        "input": input_np,
        "target": target_np,
        "var": var_np
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Homoscedastic variance (var has one less dimension)
    input_np = np.random.randn(5, 2).astype(np.float32)
    target_np = np.random.randn(5, 2).astype(np.float32)
    var_np = np.random.rand(5, 1).astype(np.float32) + 1e-6
    input_dict = {
        "full": True,
        "eps": 1e-8,
        "reduction": 'sum',
        "input": input_np,
        "target": target_np,
        "var": var_np
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 3: 1D input and target
    input_np = np.random.randn(5).astype(np.float32)
    target_np = np.random.randn(5).astype(np.float32)
    var_np = np.random.rand(5).astype(np.float32) + 1e-6
    input_dict = {
        "full": False,
        "eps": 1e-7,
        "reduction": 'none',
        "input": input_np,
        "target": target_np,
        "var": var_np
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.GaussianNLLLoss"] = GaussianNLLLoss_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.GaussianNLLLoss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.GaussianNLLLoss'.")

check_valid('torch.nn.GaussianNLLLoss', generated_inputs['torch.nn.GaussianNLLLoss'], lib="torch")
