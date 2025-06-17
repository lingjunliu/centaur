
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def square_inputs():
    list_of_inputs = []

    # Input 1: 1D float tensor
    input1 = torch.randn(5).numpy()
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D int tensor with negative values
    input2 = torch.randint(-10, 10, (3, 3)).numpy()
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: scalar tensor
    input3 = torch.tensor(-3.14).numpy()
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 3D float tensor
    input4 = torch.randn(1,2,3).numpy()
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 1D int tensor with both positive and negative values
    input5 = torch.randint(-5, 6, (10,)).numpy()
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.square"] = square_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.square' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.square'.")

check_valid('torch.square', generated_inputs['torch.square'], lib="torch")
