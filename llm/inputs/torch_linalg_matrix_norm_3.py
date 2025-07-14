
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def matrix_norm_inputs():
    list_of_inputs = []

    # Input 1
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]])
    ord = None
    dim = (0, 1)
    keepdim = False
    out = None
    dtype = np.dtype('float64')

    input_dict = {
        "input": tensor,
        "ord": ord,
        "dim": dim,
        "keepdim": keepdim,
        "out": out,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    ord = 2
    dim = (1, 2)
    keepdim = True
    out = None
    dtype = np.dtype('float32')
    input_dict = {
        "input": tensor,
        "ord": ord,
        "dim": dim,
        "keepdim": keepdim,
        "out": out,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    tensor = np.array([[-1.0, 2.0], [3.0, -4.0]])
    ord = 'fro'
    dim = (0, 1)
    keepdim = False
    out = None
    dtype = np.dtype('float64')
    input_dict = {
        "input": tensor,
        "ord": ord,
        "dim": dim,
        "keepdim": keepdim,
        "out": out,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]])
    ord = 1
    dim = (0, 1)
    keepdim = True
    out = None
    dtype = np.dtype('float32')
    input_dict = {
        "input": tensor,
        "ord": ord,
        "dim": dim,
        "keepdim": keepdim,
        "out": out,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]])
    ord = np.inf
    dim = (0, 1)
    keepdim = False
    out = None
    dtype = np.dtype('float64')
    input_dict = {
        "input": tensor,
        "ord": ord,
        "dim": dim,
        "keepdim": keepdim,
        "out": out,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    ord = -2
    dim = (1, 2)
    keepdim = True
    out = None
    dtype = np.dtype('float32')
    input_dict = {
        "input": tensor,
        "ord": ord,
        "dim": dim,
        "keepdim": keepdim,
        "out": out,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    tensor = np.array([[-1.0, 2.0], [3.0, -4.0]])
    ord = 'nuc'
    dim = (0, 1)
    keepdim = False
    out = None
    dtype = np.dtype('float64')
    input_dict = {
        "input": tensor,
        "ord": ord,
        "dim": dim,
        "keepdim": keepdim,
        "out": out,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]])
    ord = -1
    dim = (0, 1)
    keepdim = True
    out = None
    dtype = np.dtype('float32')
    input_dict = {
        "input": tensor,
        "ord": ord,
        "dim": dim,
        "keepdim": keepdim,
        "out": out,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]])
    ord = -np.inf
    dim = (0, 1)
    keepdim = False
    out = None
    dtype = np.dtype('float64')
    input_dict = {
        "input": tensor,
        "ord": ord,
        "dim": dim,
        "keepdim": keepdim,
        "out": out,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    ord = None
    dim = (1, 2)
    keepdim = True
    out = None
    dtype = np.dtype('float32')
    input_dict = {
        "input": tensor,
        "ord": ord,
        "dim": dim,
        "keepdim": keepdim,
        "out": out,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    ord = 2.0
    dim = (1, 2)
    keepdim = True
    out = None
    dtype = np.dtype('float32')
    input_dict = {
        "input": tensor,
        "ord": ord,
        "dim": dim,
        "keepdim": keepdim,
        "out": out,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.linalg.matrix_norm_3"] = matrix_norm_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.linalg.matrix_norm_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.matrix_norm_3'.")

check_valid('torch.linalg.matrix_norm', generated_inputs['torch.linalg.matrix_norm_3'], lib="torch", suffix=3)
