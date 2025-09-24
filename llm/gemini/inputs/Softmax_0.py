
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def softmax_inputs():
    list_of_inputs = []

    # Input 1: 2D tensor, dim=1
    input1 = np.random.randn(2, 3)
    input_dict1 = {"input": input1, "dim": 1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 3D tensor, dim=0
    input2 = np.random.randn(3, 4, 5)
    input_dict2 = {"input": input2, "dim": 0}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 1D tensor, dim=0
    input3 = np.random.randn(5)
    input_dict3 = {"input": input3, "dim": 0}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 4D tensor with negative values, dim=2
    input4 = np.random.randn(2, 3, 4, 5) * -1
    input_dict4 = {"input": input4, "dim": 2}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 2D tensor of integers, dim=0. Convert to float
    input5 = np.random.randint(-5, 5, size=(2, 3)).astype(np.float32)
    input_dict5 = {"input": input5, "dim": 0}
    list_of_inputs.append(copy.deepcopy(input_dict5))


    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.Softmax"] = softmax_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.Softmax' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Softmax'.")

check_valid('torch.nn.Softmax', generated_inputs['torch.nn.Softmax'], lib="torch")
