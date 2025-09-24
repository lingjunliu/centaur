
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np

def eye_inputs():
    list_of_inputs = []

    # Input 1: Basic square matrix
    n = 3
    m = 3
    out = torch.empty(n, m, dtype=torch.float32).numpy()
    dtype = torch.float32
    layout = torch.strided
    requires_grad = False
    input_dict = {
        "n": n,
        "m": m,
        "out": out,
        "dtype": dtype,
        "layout": layout,
        "requires_grad": requires_grad,
    }
    list_of_inputs.append(input_dict)

    # Input 2: Rectangular matrix (n > m)
    n = 5
    m = 2
    out = torch.empty(n, m, dtype=torch.float32).numpy()
    dtype = torch.float32
    layout = torch.strided
    requires_grad = True
    input_dict = {
        "n": n,
        "m": m,
        "out": out,
        "dtype": dtype,
        "layout": layout,
        "requires_grad": requires_grad,
    }
    list_of_inputs.append(input_dict)

    # Input 3: Rectangular matrix (n < m)
    n = 2
    m = 5
    out = torch.empty(n, m, dtype=torch.float64).numpy()
    dtype = torch.float64
    layout = torch.strided
    requires_grad = False
    input_dict = {
        "n": n,
        "m": m,
        "out": out,
        "dtype": dtype,
        "layout": layout,
        "requires_grad": requires_grad,
    }
    list_of_inputs.append(input_dict)

    # Input 4: Default m (n=m) with float32 dtype
    n = 4
    m = 4 # Changed to explicit value, removing None
    out = torch.empty(n, m, dtype=torch.float32).numpy()
    dtype = torch.float32
    layout = torch.strided
    requires_grad = True
    input_dict = {
        "n": n,
        "m": m,
        "out": out,
        "dtype": dtype,
        "layout": layout,
        "requires_grad": requires_grad,
    }
    list_of_inputs.append(input_dict)
    
    # Input 5: Small size matrix
    n = 1
    m = 1
    out = torch.empty(n, m, dtype=torch.float16).numpy()
    dtype = torch.float16
    layout = torch.strided
    requires_grad = False
    input_dict = {
        "n": n,
        "m": m,
        "out": out,
        "dtype": dtype,
        "layout": layout,
        "requires_grad": requires_grad,
    }
    list_of_inputs.append(input_dict)
    
    # Input 6: Different dtype
    n = 3
    m = 3
    out = torch.empty(n, m, dtype=torch.float32).numpy()
    dtype = torch.float32
    layout = torch.strided
    requires_grad = False
    input_dict = {
        "n": n,
        "m": m,
        "out": out,
        "dtype": dtype,
        "layout": layout,
        "requires_grad": requires_grad,
    }
    list_of_inputs.append(input_dict)

    # Input 7: Larger matrix
    n = 10
    m = 10
    out = torch.empty(n, m, dtype=torch.float32).numpy()
    dtype = torch.float32
    layout = torch.strided
    requires_grad = True
    input_dict = {
        "n": n,
        "m": m,
        "out": out,
        "dtype": dtype,
        "layout": layout,
        "requires_grad": requires_grad,
    }
    list_of_inputs.append(input_dict)

    # Input 8: n=1, different m
    n = 1
    m = 5
    out = torch.empty(n, m, dtype=torch.float32).numpy()
    dtype = torch.float32
    layout = torch.strided
    requires_grad = False
    input_dict = {
        "n": n,
        "m": m,
        "out": out,
        "dtype": dtype,
        "layout": layout,
        "requires_grad": requires_grad,
    }
    list_of_inputs.append(input_dict)

    # Input 9: m=1, different n
    n = 5
    m = 1
    out = torch.empty(n, m, dtype=torch.float64).numpy()
    dtype = torch.float64
    layout = torch.strided
    requires_grad = True
    input_dict = {
        "n": n,
        "m": m,
        "out": out,
        "dtype": dtype,
        "layout": layout,
        "requires_grad": requires_grad,
    }
    list_of_inputs.append(input_dict)
    
    # Input 10: Different dtype and requires_grad
    n = 4
    m = 4
    out = torch.empty(n, m, dtype=torch.complex64).numpy()
    dtype = torch.complex64
    layout = torch.strided
    requires_grad = False
    input_dict = {
        "n": n,
        "m": m,
        "out": out,
        "dtype": dtype,
        "layout": layout,
        "requires_grad": requires_grad,
    }
    list_of_inputs.append(input_dict)

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.eye"] = eye_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.eye' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.eye'.")

check_valid('torch.eye', generated_inputs['torch.eye'], lib="torch", suffix=0)
