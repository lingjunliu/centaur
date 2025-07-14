
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def hamming_window_inputs():
    list_of_inputs = []

    # Input 1
    window_length = np.int64(5)
    periodic = np.bool_(True)
    alpha = np.float64(0.5)
    beta = np.float64(0.5)
    dtype = torch.float32
    requires_grad = np.bool_(False)
    input_dict = {
        "window_length": int(window_length),
        "periodic": bool(periodic),
        "alpha": float(alpha),
        "beta": float(beta),
        "dtype": dtype,
        "requires_grad": bool(requires_grad)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    window_length = np.int64(10)
    periodic = np.bool_(False)
    alpha = np.float64(0.54)
    beta = np.float64(0.46)
    dtype = torch.float64
    requires_grad = np.bool_(True)
    input_dict = {
        "window_length": int(window_length),
        "periodic": bool(periodic),
        "alpha": float(alpha),
        "beta": float(beta),
        "dtype": dtype,
        "requires_grad": bool(requires_grad)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    window_length = np.int64(1)
    periodic = np.bool_(True)
    alpha = np.float64(0.0)
    beta = np.float64(1.0)
    dtype = torch.float32
    requires_grad = np.bool_(False)
    input_dict = {
        "window_length": int(window_length),
        "periodic": bool(periodic),
        "alpha": float(alpha),
        "beta": float(beta),
        "dtype": dtype,
        "requires_grad": bool(requires_grad)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    window_length = np.int64(20)
    periodic = np.bool_(False)
    alpha = np.float64(1.0)
    beta = np.float64(0.0)
    dtype = torch.float64
    requires_grad = np.bool_(True)
    input_dict = {
        "window_length": int(window_length),
        "periodic": bool(periodic),
        "alpha": float(alpha),
        "beta": float(beta),
        "dtype": dtype,
        "requires_grad": bool(requires_grad)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    window_length = np.int64(7)
    periodic = np.bool_(True)
    alpha = np.float64(0.25)
    beta = np.float64(0.75)
    dtype = torch.float32
    requires_grad = np.bool_(False)
    input_dict = {
        "window_length": int(window_length),
        "periodic": bool(periodic),
        "alpha": float(alpha),
        "beta": float(beta),
        "dtype": dtype,
        "requires_grad": bool(requires_grad)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    window_length = np.int64(30)
    periodic = np.bool_(False)
    alpha = np.float64(0.75)
    beta = np.float64(0.25)
    dtype = torch.float64
    requires_grad = np.bool_(True)
    input_dict = {
        "window_length": int(window_length),
        "periodic": bool(periodic),
        "alpha": float(alpha),
        "beta": float(beta),
        "dtype": dtype,
        "requires_grad": bool(requires_grad)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    window_length = np.int64(2)
    periodic = np.bool_(True)
    alpha = np.float64(0.3)
    beta = np.float64(0.7)
    dtype = torch.float32
    requires_grad = np.bool_(False)
    input_dict = {
        "window_length": int(window_length),
        "periodic": bool(periodic),
        "alpha": float(alpha),
        "beta": float(beta),
        "dtype": dtype,
        "requires_grad": bool(requires_grad)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    window_length = np.int64(40)
    periodic = np.bool_(False)
    alpha = np.float64(0.7)
    beta = np.float64(0.3)
    dtype = torch.float64
    requires_grad = np.bool_(True)
    input_dict = {
        "window_length": int(window_length),
        "periodic": bool(periodic),
        "alpha": float(alpha),
        "beta": float(beta),
        "dtype": dtype,
        "requires_grad": bool(requires_grad)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    window_length = np.int64(3)
    periodic = np.bool_(True)
    alpha = np.float64(0.4)
    beta = np.float64(0.6)
    dtype = torch.float32
    requires_grad = np.bool_(False)
    input_dict = {
        "window_length": int(window_length),
        "periodic": bool(periodic),
        "alpha": float(alpha),
        "beta": float(beta),
        "dtype": dtype,
        "requires_grad": bool(requires_grad)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    window_length = np.int64(50)
    periodic = np.bool_(False)
    alpha = np.float64(0.6)
    beta = np.float64(0.4)
    dtype = torch.float64
    requires_grad = np.bool_(True)
    input_dict = {
        "window_length": int(window_length),
        "periodic": bool(periodic),
        "alpha": float(alpha),
        "beta": float(beta),
        "dtype": dtype,
        "requires_grad": bool(requires_grad)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11
    window_length = np.int64(5)
    periodic = np.bool_(True)
    alpha = np.float64(0.5)
    beta = np.float64(0.5)
    dtype = np.dtype('float32')
    requires_grad = np.bool_(False)
    input_dict = {
        "window_length": int(window_length),
        "periodic": bool(periodic),
        "alpha": float(alpha),
        "beta": float(beta),
        "dtype": torch.float32,
        "requires_grad": bool(requires_grad)
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
