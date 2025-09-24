
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def blackman_window_inputs():
    list_of_inputs = []

    # Input 1
    window_length = np.int32(5)
    periodic = np.bool_(True)
    dtype = torch.float32
    layout = "strided"
    requires_grad = np.bool_(False)

    input_dict = {
        "window_length": int(window_length),
        "periodic": bool(periodic),
        "dtype": dtype,
        "layout": None,
        "requires_grad": bool(requires_grad)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    window_length = np.int64(10)
    periodic = np.bool_(False)
    dtype = torch.float64
    layout = "strided"
    requires_grad = np.bool_(True)
    input_dict = {
        "window_length": int(window_length),
        "periodic": bool(periodic),
        "dtype": dtype,
        "layout": None,
        "requires_grad": bool(requires_grad)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    window_length = np.int32(1)
    periodic = np.bool_(True)
    dtype = torch.float32
    layout = "strided"
    requires_grad = np.bool_(False)
    input_dict = {
        "window_length": int(window_length),
        "periodic": bool(periodic),
        "dtype": dtype,
        "layout": None,
        "requires_grad": bool(requires_grad)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    window_length = np.int64(20)
    periodic = np.bool_(False)
    dtype = torch.float64
    layout = "strided"
    requires_grad = np.bool_(True)
    input_dict = {
        "window_length": int(window_length),
        "periodic": bool(periodic),
        "dtype": dtype,
        "layout": None,
        "requires_grad": bool(requires_grad)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    window_length = np.int32(7)
    periodic = np.bool_(True)
    dtype = torch.float32
    layout = "strided"
    requires_grad = np.bool_(False)
    input_dict = {
        "window_length": int(window_length),
        "periodic": bool(periodic),
        "dtype": dtype,
        "layout": None,
        "requires_grad": bool(requires_grad)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    window_length = np.int64(15)
    periodic = np.bool_(False)
    dtype = torch.float64
    layout = "strided"
    requires_grad = np.bool_(True)
    input_dict = {
        "window_length": int(window_length),
        "periodic": bool(periodic),
        "dtype": dtype,
        "layout": None,
        "requires_grad": bool(requires_grad)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    window_length = np.int32(3)
    periodic = np.bool_(True)
    dtype = torch.float32
    layout = "strided"
    requires_grad = np.bool_(False)
    input_dict = {
        "window_length": int(window_length),
        "periodic": bool(periodic),
        "dtype": dtype,
        "layout": None,
        "requires_grad": bool(requires_grad)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    window_length = np.int64(25)
    periodic = np.bool_(False)
    dtype = torch.float64
    layout = "strided"
    requires_grad = np.bool_(True)
    input_dict = {
        "window_length": int(window_length),
        "periodic": bool(periodic),
        "dtype": dtype,
        "layout": None,
        "requires_grad": bool(requires_grad)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    window_length = np.int32(9)
    periodic = np.bool_(True)
    dtype = torch.float32
    layout = "strided"
    requires_grad = np.bool_(False)
    input_dict = {
        "window_length": int(window_length),
        "periodic": bool(periodic),
        "dtype": dtype,
        "layout": None,
        "requires_grad": bool(requires_grad)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    window_length = np.int64(2)
    periodic = np.bool_(False)
    dtype = torch.float64
    layout = "strided"
    requires_grad = np.bool_(True)
    input_dict = {
        "window_length": int(window_length),
        "periodic": bool(periodic),
        "dtype": dtype,
        "layout": None,
        "requires_grad": bool(requires_grad)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.blackman_window"] = blackman_window_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.blackman_window' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.blackman_window'.")

check_valid('torch.blackman_window', generated_inputs['torch.blackman_window'], lib="torch", suffix=0)
