
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import copy
import numpy as np

def block_diag_inputs():
    list_of_inputs = []

    # Input 1: A single 2D integer tensor.
    # torch.block_diag([T]) should return T.
    input_dict_1 = {
        'tensors': np.array([[1, 0], [0, 1]], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: A single 2D float tensor.
    input_dict_2 = {
        'tensors': np.array([[1.1, 2.2], [3.3, 4.4]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: A single 1D tensor. torch.block_diag iterates over its elements.
    input_dict_3 = {
        'tensors': np.array([1, 2, 3], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: A single 0D tensor (scalar).
    input_dict_4 = {
        'tensors': np.array(-10, dtype=np.int16)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: A single non-square 2D tensor.
    input_dict_5 = {
        'tensors': np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: A column vector (2D tensor).
    input_dict_6 = {
        'tensors': np.array([[1.], [2.], [3.]], dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: A row vector (2D tensor).
    input_dict_7 = {
        'tensors': np.array([[4, 5, 6, 7]], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))
    
    # Input 8: A single 2D boolean tensor.
    input_dict_8 = {
        'tensors': np.array([[True, False], [False, True]], dtype=np.bool_)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: A single 2D complex tensor.
    input_dict_9 = {
        'tensors': np.array([[1+2j, 3-4j], [0, -1j]], dtype=np.complex64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: An empty 1D tensor.
    input_dict_10 = {
        'tensors': np.array([], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: An empty 2D tensor.
    input_dict_11 = {
        'tensors': np.empty((0, 5), dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    # Input 12: Another empty 2D tensor.
    input_dict_12 = {
        'tensors': np.empty((3, 0), dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_12))

    return list_of_inputs

generated_inputs["torch.block_diag"] = block_diag_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.block_diag' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.block_diag'.")

check_valid('torch.block_diag', generated_inputs['torch.block_diag'], lib="torch", suffix=0)
