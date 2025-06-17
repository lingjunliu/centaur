
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def floor_inputs():
    list_of_inputs = []

    # Input 1: 1D float tensor with positive and negative values
    input1 = np.array([1.2, -2.5, 3.7, -4.1, 0.0], dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D float tensor with different values
    input2 = np.array([[1.1, 2.9], [-3.5, 4.0]], dtype=np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D float tensor
    input3 = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    # Input 4: scalar float
    input4 = np.array(5.6, dtype=np.float32)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 1D tensor with very small float values
    input5 = np.array([1e-8, -1e-7, 2e-9], dtype=np.float32)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: 2D tensor with large float values
    input6 = np.array([[1e8, -1e7], [2e9, -3e8]], dtype=np.float64)
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    return list_of_inputs

generated_inputs["torch.floor_"] = floor_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.floor_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.floor_'.")

check_valid('torch.floor_', generated_inputs['torch.floor_'], lib="torch")
