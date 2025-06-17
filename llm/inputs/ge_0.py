
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def ge_inputs():
    list_of_inputs = []

    # Case 1: Basic case with two tensors
    input1 = torch.tensor([[1, 2], [3, 4]]).numpy()
    input2 = torch.tensor([[1, 1], [4, 4]]).numpy()
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Comparing with a scalar
    input1 = torch.tensor([[1, 2], [3, 4]]).numpy()
    input2 = 2.0
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Using negative values
    input1 = torch.tensor([[-1, 2], [-3, 4]]).numpy()
    input2 = torch.tensor([[0, 1], [-2, 5]]).numpy()
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Broadcasting
    input1 = torch.tensor([[1, 2, 3], [4, 5, 6]]).numpy()
    input2 = torch.tensor([2]).numpy()
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Different shapes with broadcasting
    input1 = torch.tensor([[1, 2, 3], [4, 5, 6]]).numpy()
    input2 = torch.tensor([1, 5, 2]).numpy()
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: 1D tensors
    input1 = torch.tensor([1, 2, 3, 4]).numpy()
    input2 = torch.tensor([2, 1, 4, 3]).numpy()
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7: Empty tensor
    input1 = torch.tensor([]).numpy()
    input2 = torch.tensor([]).numpy()
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.ge"] = ge_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.ge' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.ge'.")

check_valid('torch.ge', generated_inputs['torch.ge'], lib="torch")
