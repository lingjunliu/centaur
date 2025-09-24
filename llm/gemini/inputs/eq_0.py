
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def torch_eq_inputs():
    list_of_inputs = []

    # Case 1: Basic integer tensors
    input1 = torch.tensor([[1, 2], [3, 4]]).numpy()
    input2 = torch.tensor([[1, 1], [4, 4]]).numpy()
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Float tensors with broadcasting
    input1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    input2 = 2.0
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Negative values and different shapes
    input1 = torch.tensor([[-1, 0, 1], [-2, 2, -3]]).numpy()
    input2 = torch.tensor([-1, 0, 1]).numpy()
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: 1D tensors
    input1 = torch.tensor([1, 2, 3, 4, 5]).numpy()
    input2 = torch.tensor([5, 4, 3, 2, 1]).numpy()
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 5: 3D tensors
    input1 = torch.randint(0, 10, (2, 3, 4)).float().numpy()
    input2 = torch.randint(0, 10, (2, 3, 4)).float().numpy()
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.eq"] = torch_eq_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.eq' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.eq'.")

check_valid('torch.eq', generated_inputs['torch.eq'], lib="torch")
