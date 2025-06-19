
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def reshape_inputs():
    list_of_inputs = []

    # Example 1: Simple reshape
    input1 = np.arange(12).reshape(3, 4)
    shape1 = (4, 3)
    input_dict1 = {"input": input1, "shape": shape1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Example 2: Using -1 to infer dimension
    input2 = np.arange(24).reshape(2, 3, 4)
    shape2 = (6, -1)
    input_dict2 = {"input": input2, "shape": shape2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Example 3: Reshape to a single dimension
    input3 = np.random.rand(2, 2, 2)
    shape3 = (-1,)
    input_dict3 = {"input": input3, "shape": shape3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Example 4: Reshape with float tensor
    input4 = np.array([[1.1, 2.2], [3.3, 4.4]])
    shape4 = (1, 4)
    input_dict4 = {"input": input4, "shape": shape4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Example 5: Reshape a 1D tensor
    input5 = np.arange(5)
    shape5 = (5, 1)
    input_dict5 = {"input": input5, "shape": shape5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Example 6: Reshape a tensor to same shape
    input6 = np.arange(6).reshape(2,3)
    shape6 = (2,3)
    input_dict6 = {"input": input6, "shape": shape6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Example 7: Reshape a zero dimension
    input7 = np.array(5)
    shape7 = (1,)
    input_dict7 = {"input": input7, "shape": shape7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.reshape"] = reshape_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.reshape' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.reshape'.")

check_valid('torch.reshape', generated_inputs['torch.reshape'], lib="torch")
