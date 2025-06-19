
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import copy
import numpy as np

def isneginf_inputs():
    list_of_inputs = []

    # Input 1: 1D tensor with negative infinity
    input_tensor = torch.tensor([-float('inf'), 1.0, 0.0, float('inf')])
    out_tensor = torch.empty_like(input_tensor, dtype=torch.bool).numpy()
    input_dict = {"input": input_tensor.numpy(), "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D tensor with negative infinity
    input_tensor = torch.tensor([[float('-inf'), 1.0], [0.0, float('inf')]])
    out_tensor = torch.empty_like(input_tensor, dtype=torch.bool).numpy()
    input_dict = {"input": input_tensor.numpy(), "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D tensor without negative infinity
    input_tensor = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    out_tensor = torch.empty_like(input_tensor, dtype=torch.bool).numpy()
    input_dict = {"input": input_tensor.numpy(), "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D tensor with only negative infinity
    input_tensor = torch.tensor([float('-inf'), float('-inf'), float('-inf')])
    out_tensor = torch.empty_like(input_tensor, dtype=torch.bool).numpy()
    input_dict = {"input": input_tensor.numpy(), "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D tensor with some negative infinities
    input_tensor = torch.tensor([[float('-inf'), 2.0, float('-inf')], [4.0, float('-inf'), 6.0]])
    out_tensor = torch.empty_like(input_tensor, dtype=torch.bool).numpy()
    input_dict = {"input": input_tensor.numpy(), "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.isneginf"] = isneginf_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.isneginf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.isneginf'.")

check_valid('torch.isneginf', generated_inputs['torch.isneginf'], lib="torch", suffix=0)
