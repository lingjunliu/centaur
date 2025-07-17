
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def matrix_norm_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    ord_val = None
    dim_val = (0, 1)
    keepdim_val = False
    out_val = np.array([])
    dtype_val = np.float32

    input_dict = {
        "input": input_tensor,
        "ord": [ord_val] if ord_val is not None else [],
        "dim": dim_val,
        "keepdim": keepdim_val,
        "out": out_val,
        "dtype": dtype_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    ord_val = 2
    dim_val = (0, 1)
    keepdim_val = True
    out_val = np.array([])
    dtype_val = np.float32

    input_dict = {
        "input": input_tensor,
        "ord": [ord_val],
        "dim": dim_val,
        "keepdim": keepdim_val,
        "out": out_val,
        "dtype": dtype_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    ord_val = 'fro'
    dim_val = (1, 2)
    keepdim_val = False
    out_val = np.array([])
    dtype_val = np.float32

    input_dict = {
        "input": input_tensor,
        "ord": [ord_val],
        "dim": dim_val,
        "keepdim": keepdim_val,
        "out": out_val,
        "dtype": dtype_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    ord_val = 'nuc'
    dim_val = (1, 2)
    keepdim_val = True
    out_val = np.array([])
    dtype_val = np.float32

    input_dict = {
        "input": input_tensor,
        "ord": [ord_val],
        "dim": dim_val,
        "keepdim": keepdim_val,
        "out": out_val,
        "dtype": dtype_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
    ord_val = np.inf
    dim_val = (0, 1)
    keepdim_val = False
    out_val = np.array([])
    dtype_val = np.float32

    input_dict = {
        "input": input_tensor,
        "ord": [ord_val],
        "dim": dim_val,
        "keepdim": keepdim_val,
        "out": out_val,
        "dtype": dtype_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
    ord_val = -np.inf
    dim_val = (0, 1)
    keepdim_val = True
    out_val = np.array([])
    dtype_val = np.float32

    input_dict = {
        "input": input_tensor,
        "ord": [ord_val],
        "dim": dim_val,
        "keepdim": keepdim_val,
        "out": out_val,
        "dtype": dtype_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    ord_val = 1
    dim_val = (1, 2)
    keepdim_val = False
    out_val = np.array([])
    dtype_val = np.float32

    input_dict = {
        "input": input_tensor,
        "ord": [ord_val],
        "dim": dim_val,
        "keepdim": keepdim_val,
        "out": out_val,
        "dtype": dtype_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    ord_val = -1
    dim_val = (1, 2)
    keepdim_val = True
    out_val = np.array([])
    dtype_val = np.float32

    input_dict = {
        "input": input_tensor,
        "ord": [ord_val],
        "dim": dim_val,
        "keepdim": keepdim_val,
        "out": out_val,
        "dtype": dtype_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    ord_val = 2
    dim_val = (0, 1)
    keepdim_val = False
    out_val = np.array([])
    dtype_val = np.float32

    input_dict = {
        "input": input_tensor,
        "ord": [ord_val],
        "dim": dim_val,
        "keepdim": keepdim_val,
        "out": out_val,
        "dtype": dtype_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    ord_val = None
    dim_val = (0, 1)
    keepdim_val = True
    out_val = np.array([])
    dtype_val = np.float32

    input_dict = {
        "input": input_tensor,
        "ord": [ord_val] if ord_val is not None else [],
        "dim": dim_val,
        "keepdim": keepdim_val,
        "out": out_val,
        "dtype": dtype_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.linalg.matrix_norm_3"] = matrix_norm_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.linalg.matrix_norm_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.matrix_norm_3'.")

check_valid('torch.linalg.matrix_norm', generated_inputs['torch.linalg.matrix_norm_3'], lib="torch", suffix=3)
