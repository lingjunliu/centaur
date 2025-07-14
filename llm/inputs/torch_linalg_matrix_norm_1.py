
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def matrix_norm_inputs():
    list_of_inputs = []

    # Input 1
    input = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    ord = None
    dim = (0, 1)
    keepdim = False
    out = np.array([])
    dtype = None

    input_dict = {
        "input": input,
        "ord": ord,
        "dim": dim,
        "keepdim": keepdim,
        "out": out,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input = np.array([[1, 2], [-3, -4]], dtype=np.float32)
    ord = 1
    dim = (0, 1)
    keepdim = True
    out = np.array([])
    dtype = None

    input_dict = {
        "input": input,
        "ord": ord,
        "dim": dim,
        "keepdim": keepdim,
        "out": out,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    ord = 2
    dim = (0, 1)
    keepdim = False
    out = np.array([])
    dtype = None

    input_dict = {
        "input": input,
        "ord": ord,
        "dim": dim,
        "keepdim": keepdim,
        "out": out,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    ord = np.inf
    dim = (0, 1)
    keepdim = True
    out = np.array([])
    dtype = None

    input_dict = {
        "input": input,
        "ord": ord,
        "dim": dim,
        "keepdim": keepdim,
        "out": out,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    ord = -np.inf
    dim = (0, 1)
    keepdim = False
    out = np.array([])
    dtype = None

    input_dict = {
        "input": input,
        "ord": ord,
        "dim": dim,
        "keepdim": keepdim,
        "out": out,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    ord = 'fro'
    dim = (0, 1)
    keepdim = True
    out = np.array([])
    dtype = None

    input_dict = {
        "input": input,
        "ord": ord,
        "dim": dim,
        "keepdim": keepdim,
        "out": out,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    ord = None
    dim = (0, 1)
    keepdim = False
    out = np.array([])
    dtype = None

    input_dict = {
        "input": input,
        "ord": ord,
        "dim": dim,
        "keepdim": keepdim,
        "out": out,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    input = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    ord = 2
    dim = (1, 2)
    keepdim = True
    out = np.array([])
    dtype = None

    input_dict = {
        "input": input,
        "ord": ord,
        "dim": dim,
        "keepdim": keepdim,
        "out": out,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    ord = -1
    dim = (0, 1)
    keepdim = False
    out = np.array([])
    dtype = None

    input_dict = {
        "input": input,
        "ord": ord,
        "dim": dim,
        "keepdim": keepdim,
        "out": out,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    ord = -2
    dim = (0, 1)
    keepdim = True
    out = np.array([])
    dtype = None

    input_dict = {
        "input": input,
        "ord": ord,
        "dim": dim,
        "keepdim": keepdim,
        "out": out,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}

def convert_to_torch(item):
    if isinstance(item, np.ndarray):
        return torch.from_numpy(item)
    elif isinstance(item, list):
        return [convert_to_torch(i) for i in item]
    elif isinstance(item, tuple):
        return tuple(convert_to_torch(list(item)))
    elif isinstance(item, type(None)):
        return None
    elif isinstance(item, np.float64):
        return torch.float32(item)
    elif isinstance(item, np.int64):
        return int(item)
    elif isinstance(item, type(torch.float32)):
        return None
    else:
        return item

def prepare_inputs(input_dict):
    new_input_dict = {}
    for key, value in input_dict.items():
        new_input_dict[key] = convert_to_torch(value)
    return new_input_dict

matrix_norm_inputs_list = matrix_norm_inputs()
new_matrix_norm_inputs_list = []

for input_dict in matrix_norm_inputs_list:
    new_input_dict = prepare_inputs(input_dict)

    # Convert numpy arrays inside 'out' to tensors with the correct dtype
    if isinstance(new_input_dict['out'], torch.Tensor) and new_input_dict['out'].numel() == 0:
        if new_input_dict['dtype'] is not None:
            new_input_dict['out'] = torch.tensor([], dtype=new_input_dict['dtype'])
        else:
            new_input_dict['out'] = torch.tensor([])

    new_matrix_norm_inputs_list.append(new_input_dict)

generated_inputs["torch.linalg.matrix_norm_1"] = new_matrix_norm_inputs_list

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.linalg.matrix_norm_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.matrix_norm_1'.")

check_valid('torch.linalg.matrix_norm', generated_inputs['torch.linalg.matrix_norm_1'], lib="torch", suffix=1)
