
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def tensordot_inputs():
    list_of_inputs = []

    # Input 1: Basic case with dims=1
    a = torch.randn(3, 4, 5).numpy()
    b = torch.randn(5, 2, 3).numpy()
    dims = 1
    out = torch.tensor([]).numpy()
    input_dict = {"a": a, "b": b, "dims": dims, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: dims=0 (dot product)
    a = torch.randn(5).numpy()
    b = torch.randn(5).numpy()
    dims = 0
    out = torch.tensor([]).numpy()
    input_dict = {"a": a, "b": b, "dims": dims, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Smaller tensors
    a = torch.randn(2, 2).numpy()
    b = torch.randn(2, 2).numpy()
    dims = 1
    out = torch.tensor([]).numpy()
    input_dict = {"a": a, "b": b, "dims": dims, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: dims = 1, different shapes
    a = torch.randn(3, 4).numpy()
    b = torch.randn(4, 5).numpy()
    dims = 1
    out = torch.tensor([]).numpy()
    input_dict = {"a": a, "b": b, "dims": dims, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: dims = 1, 3D tensors with matching dimension
    a = torch.randn(2, 3, 4).numpy()
    b = torch.randn(4, 5, 6).numpy()
    dims = 1
    out = torch.tensor([]).numpy()
    input_dict = {"a": a, "b": b, "dims": dims, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.tensordot_1"] = tensordot_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.tensordot_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.tensordot_1'.")

check_valid('torch.tensordot', generated_inputs['torch.tensordot_1'], lib="torch", suffix=1)
