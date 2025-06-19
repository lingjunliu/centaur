
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def torch_triu_inputs():
    list_of_inputs = []

    # Input 1: Basic 3x3 float tensor, diagonal = 0
    input1 = np.random.randn(3, 3).astype(np.float32)
    input_dict1 = {"input": input1, "diagonal": 0, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 4x5 int tensor, diagonal = 1
    input2 = np.random.randint(-5, 5, size=(4, 5)).astype(np.int32)
    input_dict2 = {"input": input2, "diagonal": 1, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 2x2 complex tensor, diagonal = -1
    input3 = (np.random.randn(2, 2) + 1j * np.random.randn(2, 2)).astype(np.complex64)
    input_dict3 = {"input": input3, "diagonal": -1, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 5x5 float tensor, diagonal = 2
    input4 = np.random.randn(5, 5).astype(np.float64)
    input_dict4 = {"input": input4, "diagonal": 2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 6x4 int tensor, diagonal = -2
    input5 = np.random.randint(-10, 10, size=(6, 4)).astype(np.int64)
    input_dict5 = {"input": input5, "diagonal": -2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.triu"] = torch_triu_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.triu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.triu'.")

check_valid('torch.triu', generated_inputs['torch.triu'], lib="torch")
