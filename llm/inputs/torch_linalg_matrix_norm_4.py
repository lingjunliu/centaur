
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def matrix_norm_inputs():
    list_of_inputs = []

    # Input 11 - Valid
    input_tensor = np.array([[1.0, 2.0], [3.0, 4.0]])
    ord_tensor = 2.0
    dim_tuple = (0, 1)
    keepdim_bool = False
    out_tensor = torch.tensor([])
    dtype_type = torch.float32

    input_dict = {
        "input": torch.tensor(input_tensor),
        "ord": torch.tensor(ord_tensor),
        "dim": dim_tuple,
        "keepdim": keepdim_bool,
        "out": out_tensor,
        "dtype": dtype_type
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12 - Valid
    input_tensor = np.array([[-1.0, 2.0], [3.0, -4.0]])
    ord_tensor = 1.0
    dim_tuple = (0, 1)
    keepdim_bool = True
    out_tensor = torch.tensor([])
    dtype_type = torch.float64

    input_dict = {
        "input": torch.tensor(input_tensor),
        "ord": torch.tensor(ord_tensor),
        "dim": dim_tuple,
        "keepdim": keepdim_bool,
        "out": out_tensor,
        "dtype": dtype_type
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 13 - Valid
    input_tensor = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    ord_tensor = 'fro'
    dim_tuple = (0, 1)
    keepdim_bool = False
    out_tensor = torch.tensor([])
    dtype_type = torch.float32

    input_dict = {
        "input": torch.tensor(input_tensor),
        "ord": ord_tensor,
        "dim": dim_tuple,
        "keepdim": keepdim_bool,
        "out": out_tensor,
        "dtype": dtype_type
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 14 - Valid
    input_tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    ord_tensor = np.inf
    dim_tuple = (1, 2)
    keepdim_bool = True
    out_tensor = torch.tensor([])
    dtype_type = torch.float64

    input_dict = {
        "input": torch.tensor(input_tensor),
        "ord": torch.tensor(ord_tensor),
        "dim": dim_tuple,
        "keepdim": keepdim_bool,
        "out": out_tensor,
        "dtype": dtype_type
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 15 - Valid
    input_tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    ord_tensor = -np.inf
    dim_tuple = (1, 2)
    keepdim_bool = False
    out_tensor = torch.tensor([])
    dtype_type = torch.float32

    input_dict = {
        "input": torch.tensor(input_tensor),
        "ord": torch.tensor(ord_tensor),
        "dim": dim_tuple,
        "keepdim": keepdim_bool,
        "out": out_tensor,
        "dtype": dtype_type
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 16 - Valid
    input_tensor = np.array([[1.0, 2.0], [3.0, 4.0]])
    ord_tensor = -1
    dim_tuple = (0, 1)
    keepdim_bool = True
    out_tensor = torch.tensor([])
    dtype_type = torch.float64

    input_dict = {
        "input": torch.tensor(input_tensor),
        "ord": torch.tensor(ord_tensor),
        "dim": dim_tuple,
        "keepdim": keepdim_bool,
        "out": out_tensor,
        "dtype": dtype_type
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 17 - Valid
    input_tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    ord_tensor = 2
    dim_tuple = (0, 1)
    keepdim_bool = False
    out_tensor = torch.tensor([])
    dtype_type = torch.float32

    input_dict = {
        "input": torch.tensor(input_tensor),
        "ord": torch.tensor(ord_tensor),
        "dim": dim_tuple,
        "keepdim": keepdim_bool,
        "out": out_tensor,
        "dtype": dtype_type
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 18 - Valid
    input_tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    ord_tensor = -2
    dim_tuple = (0, 2)
    keepdim_bool = True
    out_tensor = torch.tensor([])
    dtype_type = torch.float64

    input_dict = {
        "input": torch.tensor(input_tensor),
        "ord": torch.tensor(ord_tensor),
        "dim": dim_tuple,
        "keepdim": keepdim_bool,
        "out": out_tensor,
        "dtype": dtype_type
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 19 - Valid
    input_tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.complex64)
    ord_tensor = 2.0
    dim_tuple = (0, 1)
    keepdim_bool = False
    out_tensor = torch.tensor([])
    dtype_type = torch.complex64

    input_dict = {
        "input": torch.tensor(input_tensor),
        "ord": torch.tensor(ord_tensor),
        "dim": dim_tuple,
        "keepdim": keepdim_bool,
        "out": out_tensor,
        "dtype": dtype_type
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 20 - Valid
    input_tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.complex128)
    ord_tensor = 'fro'
    dim_tuple = (1, 2)
    keepdim_bool = True
    out_tensor = torch.tensor([])
    dtype_type = torch.complex128

    input_dict = {
        "input": torch.tensor(input_tensor),
        "ord": ord_tensor,
        "dim": dim_tuple,
        "keepdim": keepdim_bool,
        "out": out_tensor,
        "dtype": dtype_type
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.linalg.matrix_norm_4"] = matrix_norm_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.linalg.matrix_norm_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.matrix_norm_4'.")

check_valid('torch.linalg.matrix_norm', generated_inputs['torch.linalg.matrix_norm_4'], lib="torch", suffix=4)
