
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import copy
import numpy as np

def sparse_coo_tensor_3_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D float tensor, no grad
    input_dict_1 = {
        'size': (2, 3),
        'dtype': np.float32,
        'requires_grad': False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 2D float tensor with grad
    input_dict_2 = {
        'size': (5, 5),
        'dtype': np.float32,
        'requires_grad': True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 1D int tensor, no grad
    input_dict_3 = {
        'size': (10,),
        'dtype': np.int64,
        'requires_grad': False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: 3D double tensor with grad
    input_dict_4 = {
        'size': (2, 3, 4),
        'dtype': np.float64,
        'requires_grad': True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Large sparse tensor, half precision, with grad
    input_dict_5 = {
        'size': (100, 100),
        'dtype': np.float16,
        'requires_grad': True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Complex tensor with grad
    input_dict_6 = {
        'size': (4, 4),
        'dtype': np.complex64,
        'requires_grad': True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))
    
    # Input 7: Double complex tensor with grad
    input_dict_7 = {
        'size': (2, 2, 2),
        'dtype': np.complex128,
        'requires_grad': True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: High-dimensional tensor (4D), int, no grad
    input_dict_8 = {
        'size': (1, 2, 3, 4),
        'dtype': np.int32,
        'requires_grad': False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Zero-dimensional tensor
    input_dict_9 = {
        'size': (),
        'dtype': np.float32,
        'requires_grad': False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Unsigned integer tensor
    input_dict_10 = {
        'size': (3, 5),
        'dtype': np.uint8,
        'requires_grad': False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["torch.sparse_coo_tensor_3"] = sparse_coo_tensor_3_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.sparse_coo_tensor_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sparse_coo_tensor_3'.")

check_valid('torch.sparse_coo_tensor', generated_inputs['torch.sparse_coo_tensor_3'], lib="torch", suffix=3)
