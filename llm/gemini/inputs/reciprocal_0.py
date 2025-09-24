
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def reciprocal_inputs():
    list_of_inputs = []

    # Input 1: Float tensor with positive and negative values, 1D
    input1 = torch.randn(4).numpy()
    input_dict1 = {"input": input1, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Int tensor with positive and negative values, 2D
    input2 = torch.randint(-5, 5, (2, 3)).numpy()
    input_dict2 = {"input": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Float tensor with some values close to zero, 3D
    input3 = torch.randn(2, 2, 2) * 0.1
    input3 = input3.numpy()
    input_dict3 = {"input": input3, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Int tensor with only positive values, 1D
    input4 = torch.randint(1, 10, (5,)).numpy()
    input_dict4 = {"input": input4, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Float tensor with a large value, 1D
    input5 = torch.tensor([1000.0, -0.001, 2.0, -50.0]).numpy()
    input_dict5 = {"input": input5, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.reciprocal"] = reciprocal_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.reciprocal' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.reciprocal'.")

check_valid('torch.reciprocal', generated_inputs['torch.reciprocal'], lib="torch")
