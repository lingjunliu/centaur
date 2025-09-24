
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def softshrink_inputs():
    list_of_inputs = []

    # Test case 1: 1D tensor with positive and negative values
    input1 = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    lambd1 = 0.5
    input_dict1 = {"lambd": lambd1, "input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Test case 2: 2D tensor with a different lambda value
    input2 = np.array([[-2.0, -1.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32)
    lambd2 = 1.0
    input_dict2 = {"lambd": lambd2, "input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Test case 3: 3D tensor with lambda = 0
    input3 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    lambd3 = 0.0
    input_dict3 = {"lambd": lambd3, "input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Test case 4: 1D tensor with larger lambda value
    input4 = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    lambd4 = 2.0
    input_dict4 = {"lambd": lambd4, "input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Test case 5: Scalar input
    input6 = np.array(0.8, dtype=np.float32)
    lambd6 = 0.6
    input_dict6 = {"lambd": lambd6, "input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Test case 6: All zeros
    input7 = np.zeros((3, 3), dtype=np.float32)
    lambd7 = 0.5
    input_dict7 = {"lambd": lambd7, "input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.Softshrink"] = softshrink_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.Softshrink' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Softshrink'.")

check_valid('torch.nn.Softshrink', generated_inputs['torch.nn.Softshrink'], lib="torch")
