
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def is_floating_point_inputs():
    list_of_inputs = []

    # Input 1: Float tensor
    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Double tensor
    input2 = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Half tensor
    input3 = np.array([0.5, -1.5, 2.5], dtype=np.float16)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Higher dimensional float tensor
    input4 = np.random.rand(2, 3, 4).astype(np.float32)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Float tensor with negative values
    input5 = np.array([-1.0, -2.0, -3.0, 0.0], dtype=np.float32)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Input 6: Zero dimensional float tensor
    input6 = np.array(3.14, dtype=np.float64)
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    return list_of_inputs

generated_inputs["torch.is_floating_point"] = is_floating_point_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.is_floating_point' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.is_floating_point'.")

check_valid('torch.is_floating_point', generated_inputs['torch.is_floating_point'], lib="torch")
