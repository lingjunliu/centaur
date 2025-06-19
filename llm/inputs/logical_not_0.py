
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def logical_not_inputs():
    list_of_inputs = []

    # Input 1: 1D tensor with True/False values
    input_tensor = torch.tensor([True, False, True, True, False]).numpy()
    out_tensor = torch.tensor([]).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D tensor with boolean values
    input_tensor = torch.tensor([[True, False], [False, True]]).numpy()
    out_tensor = torch.tensor([]).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D tensor with mixed boolean values
    input_tensor = torch.tensor([[[True, False], [False, True]], [[False, True], [True, False]]]).numpy()
    out_tensor = torch.tensor([]).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: tensor with integer values (0 and 1 treated as False and True respectively)
    input_tensor = torch.tensor([0, 1, 0, 1, 1, 0], dtype=torch.int8).numpy()
    out_tensor = torch.tensor([]).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: tensor with floating point values (0.0 and non-zero treated as False and True respectively)
    input_tensor = torch.tensor([0.0, 1.0, -1.0, 0.5, 0.0]).numpy()
    out_tensor = torch.tensor([]).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.logical_not"] = logical_not_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.logical_not' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.logical_not'.")

check_valid('torch.logical_not', generated_inputs['torch.logical_not'], lib="torch", suffix=0)
