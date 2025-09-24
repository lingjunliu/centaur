
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def kaiser_window_inputs():
    list_of_inputs = []

    # Input 1
    length = 5
    periodic = True
    beta = 1.0
    dtype = torch.float32
    layout = torch.strided
    requires_grad = False
    input_dict = {
        "length": length,
        "periodic": periodic,
        "beta": beta,
        "dtype": dtype,
        "layout": layout,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    length = 10
    periodic = False
    beta = 5.0
    dtype = torch.float64
    layout = torch.strided
    requires_grad = True
    input_dict = {
        "length": length,
        "periodic": periodic,
        "beta": beta,
        "dtype": dtype,
        "layout": layout,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    length = 15
    periodic = True
    beta = 8.6
    dtype = torch.float32
    layout = torch.strided
    requires_grad = False
    input_dict = {
        "length": length,
        "periodic": periodic,
        "beta": beta,
        "dtype": dtype,
        "layout": layout,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4
    length = 20
    periodic = False
    beta = 0.0
    dtype = torch.float64
    layout = torch.strided
    requires_grad = True
    input_dict = {
        "length": length,
        "periodic": periodic,
        "beta": beta,
        "dtype": dtype,
        "layout": layout,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    length = 1
    periodic = True
    beta = 10.0
    dtype = torch.float32
    layout = torch.strided
    requires_grad = False
    input_dict = {
        "length": length,
        "periodic": periodic,
        "beta": beta,
        "dtype": dtype,
        "layout": layout,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    length = 2
    periodic = False
    beta = 2.5
    dtype = torch.float64
    layout = torch.strided
    requires_grad = True
    input_dict = {
        "length": length,
        "periodic": periodic,
        "beta": beta,
        "dtype": dtype,
        "layout": layout,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    length = 3
    periodic = True
    beta = 7.5
    dtype = torch.float32
    layout = torch.strided
    requires_grad = False
    input_dict = {
        "length": length,
        "periodic": periodic,
        "beta": beta,
        "dtype": dtype,
        "layout": layout,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    length = 4
    periodic = False
    beta = 3.3
    dtype = torch.float64
    layout = torch.strided
    requires_grad = True
    input_dict = {
        "length": length,
        "periodic": periodic,
        "beta": beta,
        "dtype": dtype,
        "layout": layout,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    length = 7
    periodic = True
    beta = 9.9
    dtype = torch.float32
    layout = torch.strided
    requires_grad = False
    input_dict = {
        "length": length,
        "periodic": periodic,
        "beta": beta,
        "dtype": dtype,
        "layout": layout,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    length = 12
    periodic = False
    beta = 6.6
    dtype = torch.float64
    layout = torch.strided
    requires_grad = True
    input_dict = {
        "length": length,
        "periodic": periodic,
        "beta": beta,
        "dtype": dtype,
        "layout": layout,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    

    return list_of_inputs

generated_inputs["torch.kaiser_window_1"] = kaiser_window_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.kaiser_window_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.kaiser_window_1'.")

check_valid('torch.kaiser_window', generated_inputs['torch.kaiser_window_1'], lib="torch", suffix=1)
