
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def eye_inputs():
    list_of_inputs = []

    # Input 1
    n = 3
    m = 3
    out = torch.empty(3, 3)
    dtype = torch.float32
    layout = torch.strided
    requires_grad = False
    input_dict = {
        "n": n,
        "m": m,
        "out": out,
        "dtype": dtype,
        "layout": layout,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    n = 5
    m = 2
    out = torch.empty(5, 2)
    dtype = torch.int64
    layout = torch.strided
    requires_grad = True
    input_dict = {
        "n": n,
        "m": m,
        "out": out,
        "dtype": dtype,
        "layout": layout,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    n = 1
    m = 7
    out = torch.empty(1, 7)
    dtype = torch.float64
    layout = torch.strided
    requires_grad = False
    input_dict = {
        "n": n,
        "m": m,
        "out": out,
        "dtype": dtype,
        "layout": layout,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    n = 4
    m = 4
    out = torch.empty(4, 4)
    dtype = torch.complex64
    layout = torch.strided
    requires_grad = True
    input_dict = {
        "n": n,
        "m": m,
        "out": out,
        "dtype": dtype,
        "layout": layout,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    n = 2
    m = 6
    out = torch.empty(2, 6)
    dtype = torch.bool
    layout = torch.strided
    requires_grad = False
    input_dict = {
        "n": n,
        "m": m,
        "out": out,
        "dtype": dtype,
        "layout": layout,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.eye"] = eye_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.eye' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.eye'.")

check_valid('torch.eye', generated_inputs['torch.eye'], lib="torch", suffix=0)
