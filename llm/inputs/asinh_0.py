
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def asinh_inputs():
    list_of_inputs = []

    # Input 1: Float tensor, 1D
    input1 = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Float tensor, 2D
    input2 = np.array([[-2.0, -1.0], [1.0, 2.0]], dtype=np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Float tensor, 3D
    input3 = np.random.rand(2, 3, 4).astype(np.float32)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Int tensor, 1D
    input4 = np.array([-2, -1, 0, 1, 2], dtype=np.int32)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5: Int tensor, 2D
    input5 = np.array([[-3, -2], [2, 3]], dtype=np.int64)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.asinh"] = asinh_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.asinh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.asinh'.")

check_valid('torch.asinh', generated_inputs['torch.asinh'], lib="torch")
