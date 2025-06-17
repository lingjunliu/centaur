
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def addmv_inputs():
    list_of_inputs = []

    # Case 1: Basic float tensors
    input_tensor = torch.randn(3)
    mat = torch.randn(3, 2)
    vec = torch.randn(2)
    input_dict = {"input": input_tensor, "mat": mat, "vec": vec, "beta": 1.0, "alpha": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Different beta and alpha
    input_tensor = torch.randn(4)
    mat = torch.randn(4, 3)
    vec = torch.randn(3)
    input_dict = {"input": input_tensor, "mat": mat, "vec": vec, "beta": 0.5, "alpha": 2.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Negative values
    input_tensor = torch.randn(5)
    mat = torch.randn(5, 4)
    vec = torch.randn(4)
    input_dict = {"input": input_tensor, "mat": mat, "vec": vec, "beta": -1.0, "alpha": -0.5}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 4: Larger tensors
    input_tensor = torch.randn(10)
    mat = torch.randn(10, 8)
    vec = torch.randn(8)
    input_dict = {"input": input_tensor, "mat": mat, "vec": vec, "beta": 0.0, "alpha": 1.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.addmv_"] = addmv_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.addmv_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.addmv_'.")

check_valid('torch.addmv_', generated_inputs['torch.addmv_'], lib="torch")
