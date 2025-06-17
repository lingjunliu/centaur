
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy
import math

def torch_exp_inputs():
    list_of_inputs = []

    # Example 1: 1D tensor with positive and negative floats
    input1 = np.array([-1.0, 0.0, 1.0, 2.0, -2.0], dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Example 2: 2D tensor with floats
    input2 = np.array([[1.5, 2.5], [-3.5, 4.5]], dtype=np.float32)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Example 3: 3D tensor with floats
    input3 = np.random.rand(2, 3, 4).astype(np.float32)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Example 4: 1D tensor with log values
    input4 = np.array([0.0, math.log(2.0), math.log(5.0)], dtype=np.float32)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.exp"] = torch_exp_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.exp' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.exp'.")

check_valid('torch.exp', generated_inputs['torch.exp'], lib="torch")
