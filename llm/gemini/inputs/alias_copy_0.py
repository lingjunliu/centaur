
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def alias_copy_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor
    input1 = torch.randn(3, 4).numpy()
    input_dict1 = {"self": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Integer tensor
    input2 = torch.randint(-5, 5, (2, 2)).numpy()
    input_dict2 = {"self": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Complex tensor
    input3 = torch.randn(2, 3, dtype=torch.complex64).numpy()
    input_dict3 = {"self": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 1D tensor
    input4 = torch.randn(5).numpy()
    input_dict4 = {"self": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 3D tensor
    input5 = torch.randn(2, 3, 4).numpy()
    input_dict5 = {"self": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Bool Tensor
    input6 = torch.randint(0, 2, (3, 3)).bool().numpy()
    input_dict6 = {"self": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Tensor with negative values
    input7 = torch.randn(4, 4) * -1.0
    input7 = input7.numpy()
    input_dict7 = {"self": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs

generated_inputs["torch.alias_copy"] = alias_copy_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.alias_copy' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.alias_copy'.")

check_valid('torch.alias_copy', generated_inputs['torch.alias_copy'], lib="torch")
