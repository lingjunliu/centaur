
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def matrix_norm_inputs():
    list_of_inputs = []

    # Input 1
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    ord = 'fro'
    dim = (0, 1)
    keepdim = False
    out = np.array([], dtype=np.float32)
    dtype = torch.float32
    input_dict = {"input": tensor, "ord": ord, "dim": dim, "keepdim": keepdim, "out": torch.tensor(out, dtype=torch.float32), "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    ord = 2
    dim = (0, 1)
    keepdim = True
    out = np.array([], dtype=np.float32)
    dtype = torch.float32
    input_dict = {"input": tensor, "ord": ord, "dim": dim, "keepdim": keepdim, "out": torch.tensor(out, dtype=torch.float32), "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    tensor = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    ord = 1
    dim = (0, 1)
    keepdim = False
    out = np.array([], dtype=np.float32)
    dtype = torch.float32
    input_dict = {"input": tensor, "ord": ord, "dim": dim, "keepdim": keepdim, "out": torch.tensor(out, dtype=torch.float32), "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    tensor = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    ord = -1
    dim = (0, 1)
    keepdim = True
    out = np.array([], dtype=np.float32)
    dtype = torch.float32
    input_dict = {"input": tensor, "ord": ord, "dim": dim, "keepdim": keepdim, "out": torch.tensor(out, dtype=torch.float32), "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    tensor = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    ord = 2
    dim = (0, 1)
    keepdim = False
    out = np.array([], dtype=np.float32)
    dtype = torch.float32
    input_dict = {"input": tensor, "ord": ord, "dim": dim, "keepdim": keepdim, "out": torch.tensor(out, dtype=torch.float32), "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    tensor = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    ord = -2
    dim = (0, 1)
    keepdim = True
    out = np.array([], dtype=np.float32)
    dtype = torch.float32
    input_dict = {"input": tensor, "ord": ord, "dim": dim, "keepdim": keepdim, "out": torch.tensor(out, dtype=torch.float32), "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    ord = 'fro'
    dim = (1, 2)
    keepdim = False
    out = np.array([], dtype=np.float32)
    dtype = torch.float32
    input_dict = {"input": tensor, "ord": ord, "dim": dim, "keepdim": keepdim, "out": torch.tensor(out, dtype=torch.float32), "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    ord = 2
    dim = (1, 2)
    keepdim = True
    out = np.array([], dtype=np.float32)
    dtype = torch.float32
    input_dict = {"input": tensor, "ord": ord, "dim": dim, "keepdim": keepdim, "out": torch.tensor(out, dtype=torch.float32), "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    ord = 1
    dim = (1, 2)
    keepdim = False
    out = np.array([], dtype=np.float32)
    dtype = torch.float32
    input_dict = {"input": tensor, "ord": ord, "dim": dim, "keepdim": keepdim, "out": torch.tensor(out, dtype=torch.float32), "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    ord = -1
    dim = (1, 2)
    keepdim = True
    out = np.array([], dtype=np.float32)
    dtype = torch.float32
    input_dict = {"input": tensor, "ord": ord, "dim": dim, "keepdim": keepdim, "out": torch.tensor(out, dtype=torch.float32), "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    ord = 'fro'
    dim = (-2, -1)
    keepdim = False
    out = np.array([], dtype=np.float32)
    dtype = torch.float32
    input_dict = {"input": tensor, "ord": ord, "dim": dim, "keepdim": keepdim, "out": torch.tensor(out, dtype=torch.float32), "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.linalg.matrix_norm_2"] = matrix_norm_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.linalg.matrix_norm_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.matrix_norm_2'.")

check_valid('torch.linalg.matrix_norm', generated_inputs['torch.linalg.matrix_norm_2'], lib="torch", suffix=2)
