
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def acos_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D tensor with values between -1 and 1
    input1 = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D tensor with float32 data type
    input2 = np.array([[-0.8, 0.2], [0.9, -0.1]], dtype=np.float32)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D tensor with negative and positive values
    input3 = np.array([[[0.1, -0.2], [0.3, -0.4]], [[0.5, -0.6], [0.7, -0.8]]], dtype=np.float32)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Single element tensor (scalar)
    input4 = np.array(0.6, dtype=np.float32)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Larger tensor with more varied values
    input5 = np.array([[-0.9, 0.7, -0.3], [0.5, -0.2, 0.8], [-0.1, 0.4, -0.6]], dtype=np.float32)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Input 6: Test with a different range
    input6 = np.array([-0.99, -0.75, -0.25, 0.25, 0.75, 0.99], dtype=np.float32)
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    return list_of_inputs

generated_inputs["torch.acos"] = acos_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.acos' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.acos'.")

check_valid('torch.acos', generated_inputs['torch.acos'], lib="torch")
