
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def kaiser_window_inputs():
    list_of_inputs = []

    # Input 1, valid
    length = np.int32(5)
    beta = np.float32(1.0)
    periodic = np.bool_(True)
    dtype = torch.float32
    layout = 'strided'
    requires_grad = np.bool_(False)

    input_dict = {
        "length": int(length),
        "periodic": bool(periodic),
        "beta": float(beta),
        "dtype": dtype,
        "layout": layout,
        "requires_grad": bool(requires_grad)
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid
    length = np.int32(10)
    beta = np.float32(5.0)
    periodic = np.bool_(False)
    dtype = torch.float64
    layout = 'strided'
    requires_grad = np.bool_(True)

    input_dict = {
        "length": int(length),
        "periodic": bool(periodic),
        "beta": float(beta),
        "dtype": dtype,
        "layout": layout,
        "requires_grad": bool(requires_grad)
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid
    length = np.int32(1)
    beta = np.float32(0.0)
    periodic = np.bool_(True)
    dtype = torch.float16
    layout = 'strided'
    requires_grad = np.bool_(False)

    input_dict = {
        "length": int(length),
        "periodic": bool(periodic),
        "beta": float(beta),
        "dtype": dtype,
        "layout": layout,
        "requires_grad": bool(requires_grad)
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    length = np.int32(20)
    beta = np.float32(8.6)
    periodic = np.bool_(False)
    dtype = torch.float32
    layout = 'strided'
    requires_grad = np.bool_(True)

    input_dict = {
        "length": int(length),
        "periodic": bool(periodic),
        "beta": float(beta),
        "dtype": dtype,
        "layout": layout,
        "requires_grad": bool(requires_grad)
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid
    length = np.int32(7)
    beta = np.float32(3.2)
    periodic = np.bool_(True)
    dtype = torch.float64
    layout = 'strided'
    requires_grad = np.bool_(False)

    input_dict = {
        "length": int(length),
        "periodic": bool(periodic),
        "beta": float(beta),
        "dtype": dtype,
        "layout": layout,
        "requires_grad": bool(requires_grad)
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, valid
    length = np.int32(32)
    beta = np.float32(10.0)
    periodic = np.bool_(False)
    dtype = torch.float16
    layout = 'strided'
    requires_grad = np.bool_(True)

    input_dict = {
        "length": int(length),
        "periodic": bool(periodic),
        "beta": float(beta),
        "dtype": dtype,
        "layout": layout,
        "requires_grad": bool(requires_grad)
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    length = np.int32(64)
    beta = np.float32(2.5)
    periodic = np.bool_(True)
    dtype = torch.float32
    layout = 'strided'
    requires_grad = np.bool_(False)

    input_dict = {
        "length": int(length),
        "periodic": bool(periodic),
        "beta": float(beta),
        "dtype": dtype,
        "layout": layout,
        "requires_grad": bool(requires_grad)
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8, valid
    length = np.int32(128)
    beta = np.float32(7.8)
    periodic = np.bool_(False)
    dtype = torch.float64
    layout = 'strided'
    requires_grad = np.bool_(True)

    input_dict = {
        "length": int(length),
        "periodic": bool(periodic),
        "beta": float(beta),
        "dtype": dtype,
        "layout": layout,
        "requires_grad": bool(requires_grad)
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, valid
    length = np.int32(256)
    beta = np.float32(0.1)
    periodic = np.bool_(True)
    dtype = torch.float16
    layout = 'strided'
    requires_grad = np.bool_(False)

    input_dict = {
        "length": int(length),
        "periodic": bool(periodic),
        "beta": float(beta),
        "dtype": dtype,
        "layout": layout,
        "requires_grad": bool(requires_grad)
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10, valid
    length = np.int32(512)
    beta = np.float32(9.9)
    periodic = np.bool_(False)
    dtype = torch.float32
    layout = 'strided'
    requires_grad = np.bool_(True)

    input_dict = {
        "length": int(length),
        "periodic": bool(periodic),
        "beta": float(beta),
        "dtype": dtype,
        "layout": layout,
        "requires_grad": bool(requires_grad)
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.kaiser_window_2"] = kaiser_window_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.kaiser_window_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.kaiser_window_2'.")

check_valid('torch.kaiser_window', generated_inputs['torch.kaiser_window_2'], lib="torch", suffix=2)
