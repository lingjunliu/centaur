
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def SELU_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D float tensor
    input1 = np.random.randn(5).astype(np.float32)
    input_dict1 = {"input": input1, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D float tensor with negative values, inplace=True
    input2 = np.random.randn(3, 4).astype(np.float32)
    input_dict2 = {"input": input2, "inplace": True}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D float tensor
    input3 = np.random.randn(2, 3, 5).astype(np.float32)
    input_dict3 = {"input": input3, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Scalar float
    input4 = np.array(3.14).astype(np.float32)
    input_dict4 = {"input": input4, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 4D float tensor
    input5 = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_dict5 = {"input": input5, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.SELU"] = SELU_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.SELU' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.SELU'.")

check_valid('torch.nn.SELU', generated_inputs['torch.nn.SELU'], lib="torch")
