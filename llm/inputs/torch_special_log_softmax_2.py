
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def log_softmax_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([1.0, 2.0, 3.0])
    dim = 0
    dtype = torch.float32
    input_dict = {"input": input_tensor, "dim": dim, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[1.0, 2.0], [3.0, 4.0]])
    dim = 1
    dtype = torch.float64
    input_dict = {"input": input_tensor, "dim": dim, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([[-1.0, -2.0], [-3.0, -4.0]])
    dim = 0
    dtype = torch.float32
    input_dict = {"input": input_tensor, "dim": dim, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([1, 2, 3], dtype=np.int32)
    dim = 0
    dtype = torch.float64
    input_dict = {"input": input_tensor, "dim": dim, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64)
    dim = 2
    dtype = torch.float32
    input_dict = {"input": input_tensor, "dim": dim, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array([1.0, 2.0, 3.0, float('inf')])
    dim = 0
    dtype = torch.float64
    input_dict = {"input": input_tensor, "dim": dim, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array([1.0, 2.0, 3.0, float('-inf')])
    dim = 0
    dtype = torch.float32
    input_dict = {"input": input_tensor, "dim": dim, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array([1.0, 2.0, 3.0, float('nan')])
    dim = 0
    dtype = torch.float64
    input_dict = {"input": input_tensor, "dim": dim, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    dim = 1
    dtype = torch.float32
    input_dict = {"input": input_tensor, "dim": dim, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64)
    dim = 0
    dtype = torch.float64
    input_dict = {"input": input_tensor, "dim": dim, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.special.log_softmax_2"] = log_softmax_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.special.log_softmax_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.log_softmax_2'.")

check_valid('torch.special.log_softmax', generated_inputs['torch.special.log_softmax_2'], lib="torch", suffix=2)
