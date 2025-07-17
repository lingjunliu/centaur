
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def hamming_window_inputs():
    list_of_inputs = []

    # Input 1
    window_length = int(5)
    periodic = bool(True)
    alpha = float(0.5)
    beta = float(0.4)
    dtype = torch.float32
    layout = torch.strided
    requires_grad = bool(False)

    input_dict = {
        "window_length": window_length,
        "periodic": periodic,
        "alpha": alpha,
        "beta": beta,
        "dtype": dtype,
        "layout": layout,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    window_length = int(10)
    periodic = bool(False)
    alpha = float(0.54)
    beta = float(0.46)
    dtype = torch.float64
    layout = torch.strided
    requires_grad = bool(True)

    input_dict = {
        "window_length": window_length,
        "periodic": periodic,
        "alpha": alpha,
        "beta": beta,
        "dtype": dtype,
        "layout": layout,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    window_length = int(1)
    periodic = bool(True)
    alpha = float(0.0)
    beta = float(1.0)
    dtype = torch.float32
    layout = torch.strided
    requires_grad = bool(False)

    input_dict = {
        "window_length": window_length,
        "periodic": periodic,
        "alpha": alpha,
        "beta": beta,
        "dtype": dtype,
        "layout": layout,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    window_length = int(20)
    periodic = bool(False)
    alpha = float(1.0)
    beta = float(0.0)
    dtype = torch.float64
    layout = torch.strided
    requires_grad = bool(True)

    input_dict = {
        "window_length": window_length,
        "periodic": periodic,
        "alpha": alpha,
        "beta": beta,
        "dtype": dtype,
        "layout": layout,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    window_length = int(7)
    periodic = bool(True)
    alpha = float(0.25)
    beta = float(0.75)
    dtype = torch.float32
    layout = torch.strided
    requires_grad = bool(False)

    input_dict = {
        "window_length": window_length,
        "periodic": periodic,
        "alpha": alpha,
        "beta": beta,
        "dtype": dtype,
        "layout": layout,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    window_length = int(15)
    periodic = bool(False)
    alpha = float(0.75)
    beta = float(0.25)
    dtype = torch.float64
    layout = torch.strided
    requires_grad = bool(True)

    input_dict = {
        "window_length": window_length,
        "periodic": periodic,
        "alpha": alpha,
        "beta": beta,
        "dtype": dtype,
        "layout": layout,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    window_length = int(3)
    periodic = bool(True)
    alpha = float(0.33)
    beta = float(0.67)
    dtype = torch.float32
    layout = torch.strided
    requires_grad = bool(False)

    input_dict = {
        "window_length": window_length,
        "periodic": periodic,
        "alpha": alpha,
        "beta": beta,
        "dtype": dtype,
        "layout": layout,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    window_length = int(30)
    periodic = bool(False)
    alpha = float(0.67)
    beta = float(0.33)
    dtype = torch.float64
    layout = torch.strided
    requires_grad = bool(True)

    input_dict = {
        "window_length": window_length,
        "periodic": periodic,
        "alpha": alpha,
        "beta": beta,
        "dtype": dtype,
        "layout": layout,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    window_length = int(2)
    periodic = bool(True)
    alpha = float(0.6)
    beta = float(0.4)
    dtype = torch.float32
    layout = torch.strided
    requires_grad = bool(False)

    input_dict = {
        "window_length": window_length,
        "periodic": periodic,
        "alpha": alpha,
        "beta": beta,
        "dtype": dtype,
        "layout": layout,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    window_length = int(12)
    periodic = bool(False)
    alpha = float(0.4)
    beta = float(0.6)
    dtype = torch.float64
    layout = torch.strided
    requires_grad = bool(True)

    input_dict = {
        "window_length": window_length,
        "periodic": periodic,
        "alpha": alpha,
        "beta": beta,
        "dtype": dtype,
        "layout": layout,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11 - Without alpha and beta
    window_length = int(12)
    periodic = bool(False)
    dtype = torch.float64
    layout = torch.strided
    requires_grad = bool(True)

    input_dict = {
        "window_length": window_length,
        "periodic": periodic,
        "dtype": dtype,
        "layout": layout,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12 - Without periodic, alpha and beta
    window_length = int(12)
    dtype = torch.float64
    layout = torch.strided
    requires_grad = bool(True)

    input_dict = {
        "window_length": window_length,
        "dtype": dtype,
        "layout": layout,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 13 - Without periodic, alpha and beta and layout
    window_length = int(12)
    dtype = torch.float64
    requires_grad = bool(True)

    input_dict = {
        "window_length": window_length,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.hamming_window"] = hamming_window_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.hamming_window' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.hamming_window'.")

check_valid('torch.hamming_window', generated_inputs['torch.hamming_window'], lib="torch", suffix=0)
