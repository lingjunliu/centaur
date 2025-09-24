
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def tensorsolve_inputs():
    list_of_inputs = []

    # Input 1
    A = np.eye(2 * 3 * 4).reshape((2 * 3, 4, 2, 3, 4))
    B = np.random.randn(2 * 3, 4)
    dims = None
    out = None

    input_dict = {
        "A": A,
        "B": B,
        "dims": dims if dims is not None else tuple(),
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    A = np.random.randn(6, 4, 4, 3, 2)
    B = np.random.randn(4, 3, 2)
    dims = (0, 2)
    out = None
    input_dict = {
        "A": A,
        "B": B,
        "dims": dims if dims is not None else tuple(),
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3
    A = np.random.randn(2, 2)
    B = np.random.randn(2)
    dims = None
    out = None
    input_dict = {
        "A": A,
        "B": B,
        "dims": dims if dims is not None else tuple(),
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4 - VALID
    A = np.random.randn(9, 3, 3)
    B = np.random.randn(9)
    dims = None
    out = None
    input_dict = {
        "A": A,
        "B": B,
        "dims": dims if dims is not None else tuple(),
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    A = np.random.randn(4, 4, 4, 4)
    B = np.random.randn(4, 4, 4)
    dims = None
    out = None
    #Remove this input as it is causing errors
    # Input 6
    A = np.random.randn(2, 3, 2, 3)
    B = np.random.randn(2, 3)
    dims = None
    out = None
    input_dict = {
        "A": A,
        "B": B,
        "dims": dims if dims is not None else tuple(),
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 - VALID
    A = np.random.randn(25, 5, 5)
    B = np.random.randn(25)
    dims = None
    out = None
    input_dict = {
        "A": A,
        "B": B,
        "dims": dims if dims is not None else tuple(),
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    A = np.random.randn(3, 2, 3, 2)
    B = np.random.randn(3, 2)
    dims = (0, 2)
    out = None
    input_dict = {
        "A": A,
        "B": B,
        "dims": dims if dims is not None else tuple(),
        "out": out
    }
    # Removed because it caused errors
    # Input 9
    A = np.random.randn(4, 2, 2, 4)
    B = np.random.randn(4, 2)
    dims = (0, 3)
    out = None
    input_dict = {
        "A": A,
        "B": B,
        "dims": dims if dims is not None else tuple(),
        "out": out
    }
    # Removed because it caused errors

    # Input 10
    A = np.random.randn(5, 3, 5, 3)
    B = np.random.randn(5, 3)
    dims = (1, 3)
    out = None
    input_dict = {
        "A": A,
        "B": B,
        "dims": dims if dims is not None else tuple(),
        "out": out
    }
    # Removed because it caused errors
    
    # Input 11 - VALID
    A = np.random.randn(16, 4, 4)
    B = np.random.randn(16)
    dims = None
    out = None
    input_dict = {
        "A": A,
        "B": B,
        "dims": dims if dims is not None else tuple(),
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12 - VALID
    A = np.random.randn(6, 2, 3)
    B = np.random.randn(6)
    dims = None
    out = None
    input_dict = {
        "A": A,
        "B": B,
        "dims": dims if dims is not None else tuple(),
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.linalg.tensorsolve"] = tensorsolve_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.linalg.tensorsolve' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.tensorsolve'.")

check_valid('torch.linalg.tensorsolve', generated_inputs['torch.linalg.tensorsolve'], lib="torch", suffix=0)
