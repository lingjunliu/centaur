
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def clamp_max_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor
    input1 = np.array([-1.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    max_val1 = 2.0
    input_dict1 = {"input": input1, "max": max_val1, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Integer tensor with negative max
    input2 = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=np.int32)
    max_val2 = -1.0
    input_dict2 = {"input": input2, "max": max_val2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 2D float tensor
    input3 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    max_val3 = 3.0
    input_dict3 = {"input": input3, "max": max_val3, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 3D integer tensor
    input4 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64)
    max_val4 = 5.0
    input_dict4 = {"input": input4, "max": max_val4, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Large float tensor
    input5 = np.random.randn(100).astype(np.float32)
    max_val5 = 0.5
    input_dict5 = {"input": input5, "max": max_val5, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Input 6: Float tensor with a larger max value
    input6 = np.array([-5.0, -2.0, 0.0, 3.0, 7.0], dtype=np.float32)
    max_val6 = 10.0
    input_dict6 = {"input": input6, "max": max_val6, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.clamp_max"] = clamp_max_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.clamp_max' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.clamp_max'.")

check_valid('torch.clamp_max', generated_inputs['torch.clamp_max'], lib="torch")
