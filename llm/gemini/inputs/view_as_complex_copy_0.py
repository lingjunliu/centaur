
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def view_as_complex_copy_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D float tensor
    input1 = torch.randn(2, 2, 2).numpy()
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 3D float tensor
    input2 = torch.randn(2, 3, 2, 2).numpy()
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 4D float tensor
    input3 = torch.randn(2, 2, 2, 2, 2).numpy()
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    # Input 4: 1D float tensor
    input4 = torch.randn(4, 2).numpy()
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Different shape
    input5 = torch.randn(5, 1, 2).numpy()
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.view_as_complex_copy"] = view_as_complex_copy_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.view_as_complex_copy' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.view_as_complex_copy'.")

check_valid('torch.view_as_complex_copy', generated_inputs['torch.view_as_complex_copy'], lib="torch")
