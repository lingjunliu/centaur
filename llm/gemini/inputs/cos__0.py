
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def cos__inputs():
    list_of_inputs = []

    # Input 1: Float tensor with positive values
    input1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Float tensor with negative values
    input2 = np.array([-0.5, -1.0, -1.5, -2.0], dtype=np.float32)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Multi-dimensional float tensor
    input3 = np.array([[0.0, 0.5], [1.0, 1.5]], dtype=np.float64)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Float tensor with large values
    input4 = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Float tensor with values close to pi/2 and pi
    input5 = np.array([np.pi/2 - 0.1, np.pi/2, np.pi/2 + 0.1, np.pi - 0.1, np.pi, np.pi + 0.1], dtype=np.float32)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: 3D Float tensor
    input6 = np.random.rand(2, 3, 4).astype(np.float32)
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: Empty array
    input7 = np.array([], dtype=np.float32)
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs["torch.cos_"] = cos__inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.cos_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.cos_'.")

check_valid('torch.cos_', generated_inputs['torch.cos_'], lib="torch")
