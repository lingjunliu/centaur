
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def hamming_window_inputs():
    list_of_inputs = []

    # Input 1
    window_length = np.int32(5)
    periodic = np.bool_(True)
    alpha = np.float64(0.5)
    beta = np.float32(0.2)
    dtype = torch.float32
    layout = 'strided'
    requires_grad = np.bool_(False)
    
    input_dict = {
        "window_length": int(window_length),
        "periodic": bool(periodic),
        "alpha": float(alpha),
        "beta": float(beta),
        "dtype": dtype,
        "layout": layout,
        "requires_grad": bool(requires_grad)
    }
    
    kwargs = {k: input_dict[k] for k in ("dtype", "layout", "requires_grad")}
    args = [input_dict["window_length"], input_dict["periodic"], input_dict["alpha"], input_dict["beta"]]
    
    final_dict = {"window_length": int(window_length), "periodic": bool(periodic), "alpha": float(alpha), "beta": float(beta), "dtype": dtype, "layout": layout, "requires_grad": bool(requires_grad)}
    list_of_inputs.append(copy.deepcopy(final_dict))

    # Input 2
    window_length = np.int64(10)
    periodic = np.bool_(False)
    alpha = np.float16(0.2)
    beta = np.float64(0.8)
    dtype = torch.float64
    layout = 'strided'
    requires_grad = np.bool_(True)

    input_dict = {
        "window_length": int(window_length),
        "periodic": bool(periodic),
        "alpha": float(alpha),
        "beta": float(beta),
        "dtype": dtype,
        "layout": layout,
        "requires_grad": bool(requires_grad)
    }
    
    final_dict = {"window_length": int(window_length), "periodic": bool(periodic), "alpha": float(alpha), "beta": float(beta), "dtype": dtype, "layout": layout, "requires_grad": bool(requires_grad)}
    list_of_inputs.append(copy.deepcopy(final_dict))

    # Input 3
    window_length = np.int32(1)
    periodic = np.bool_(True)
    alpha = np.float32(0.0)
    beta = np.float32(1.0)
    dtype = torch.float32
    layout = 'strided'
    requires_grad = np.bool_(False)
    
    input_dict = {
        "window_length": int(window_length),
        "periodic": bool(periodic),
        "alpha": float(alpha),
        "beta": float(beta),
        "dtype": dtype,
        "layout": layout,
        "requires_grad": bool(requires_grad)
    }
    
    final_dict = {"window_length": int(window_length), "periodic": bool(periodic), "alpha": float(alpha), "beta": float(beta), "dtype": dtype, "layout": layout, "requires_grad": bool(requires_grad)}
    list_of_inputs.append(copy.deepcopy(final_dict))

    # Input 4
    window_length = np.int32(20)
    periodic = np.bool_(False)
    alpha = np.float32(1.0)
    beta = np.float32(0.0)
    dtype = torch.float32
    layout = 'strided'
    requires_grad = np.bool_(True)
    
    input_dict = {
        "window_length": int(window_length),
        "periodic": bool(periodic),
        "alpha": float(alpha),
        "beta": float(beta),
        "dtype": dtype,
        "layout": layout,
        "requires_grad": bool(requires_grad)
    }
    
    final_dict = {"window_length": int(window_length), "periodic": bool(periodic), "alpha": float(alpha), "beta": float(beta), "dtype": dtype, "layout": layout, "requires_grad": bool(requires_grad)}
    list_of_inputs.append(copy.deepcopy(final_dict))

    # Input 5
    window_length = np.int32(7)
    periodic = np.bool_(True)
    alpha = np.float32(0.7)
    beta = np.float32(0.3)
    dtype = torch.float64
    layout = 'strided'
    requires_grad = np.bool_(False)

    input_dict = {
        "window_length": int(window_length),
        "periodic": bool(periodic),
        "alpha": float(alpha),
        "beta": float(beta),
        "dtype": dtype,
        "layout": layout,
        "requires_grad": bool(requires_grad)
    }
    
    final_dict = {"window_length": int(window_length), "periodic": bool(periodic), "alpha": float(alpha), "beta": float(beta), "dtype": dtype, "layout": layout, "requires_grad": bool(requires_grad)}
    list_of_inputs.append(copy.deepcopy(final_dict))

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
