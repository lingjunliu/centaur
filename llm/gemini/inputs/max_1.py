
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def torch_max_inputs():
    list_of_inputs = []

    # Case 1: 1D float tensor
    input1 = torch.randn(5).numpy()
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Case 2: 2D int tensor with negative values
    input2 = torch.randint(-10, 10, (3, 4)).numpy()
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Case 3: 3D float tensor
    input3 = torch.randn(2, 3, 4).numpy()
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Case 4: 4D tensor with some large values
    input4 = torch.randn(1, 2, 3, 3) * 1000
    input4 = input4.numpy()
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Case 5: Tensor with all same values
    input5 = torch.full((2, 2), 5.0).numpy()
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Case 6: 2D tensor with only negative values
    input6 = torch.rand(2, 3) * -1.0
    input6 = input6.numpy()
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Case 7: 1D int tensor
    input7 = torch.randint(0, 10, (5,)).numpy()
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs

generated_inputs["torch.max_1"] = torch_max_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.max_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.max_1'.")

check_valid('torch.max', generated_inputs['torch.max_1'], lib="torch")
