
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def asarray_inputs():
    list_of_inputs = []

    # Input 1: NumPy array of integers
    input_np = np.array([1, 2, 3, 4, 5])
    input_dict = {
        "obj": input_np,
        "dtype": None,
        "copy": None,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: NumPy array of floats with specified dtype
    input_np = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
    input_dict = {
        "obj": input_np,
        "dtype": torch.float64,
        "copy": True,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: NumPy array of complex numbers
    input_np = np.array([1+1j, 2+2j, 3+3j])
    input_dict = {
        "obj": input_np,
        "dtype": None,
        "copy": False,
        "requires_grad": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: NumPy array with negative values
    input_np = np.array([-1.0, -2.0, 0.0, 1.0, 2.0])
    input_dict = {
        "obj": input_np,
        "dtype": None,
        "copy": None,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multi-dimensional NumPy array
    input_np = np.array([[1.0, 2.0], [3.0, 4.0]])
    input_dict = {
        "obj": input_np,
        "dtype": None,
        "copy": True,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.asarray"] = asarray_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.asarray' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.asarray'.")

check_valid('torch.asarray', generated_inputs['torch.asarray'], lib="torch")
