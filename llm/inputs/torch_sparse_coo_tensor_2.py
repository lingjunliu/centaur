
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import copy
import numpy as np

def torch_sparse_coo_tensor_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D float32 sparse tensor
    input_dict_1 = {
        'indices': np.array([[0, 1, 1], [2, 0, 2]], dtype=np.int64),
        'values': np.array([3.0, 4.0, 5.0], dtype=np.float32),
        'size': (2, 3),
        'dtype': torch.float32,
        'requires_grad': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 2D int32 sparse tensor
    input_dict_2 = {
        'indices': np.array([[0, 2], [1, 3]], dtype=np.int64),
        'values': np.array([10, 20], dtype=np.int32),
        'size': (4, 4),
        'dtype': torch.int32,
        'requires_grad': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 1D sparse tensor with requires_grad=True
    input_dict_3 = {
        'indices': np.array([[0, 4, 9]], dtype=np.int64),
        'values': np.array([-1.5, 2.5, -3.0], dtype=np.float64),
        'size': (10,),
        'dtype': torch.float64,
        'requires_grad': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))
    
    # Input 4: 3D int64 sparse tensor
    input_dict_4 = {
        'indices': np.array([[0, 1, 0, 2], [1, 0, 1, 2], [2, 1, 2, 0]], dtype=np.int64),
        'values': np.array([10, 20, 30, 40], dtype=np.int64),
        'size': (3, 3, 3),
        'dtype': torch.int64,
        'requires_grad': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Empty sparse tensor
    input_dict_5 = {
        'indices': np.empty((2, 0), dtype=np.int64),
        'values': np.array([], dtype=np.float32),
        'size': (5, 5),
        'dtype': torch.float32,
        'requires_grad': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Uncoalesced tensor (duplicate indices)
    input_dict_6 = {
        'indices': np.array([[0, 1, 0], [2, 2, 2]], dtype=np.int64),
        'values': np.array([3, 4, 5], dtype=np.int32),
        'size': (2, 3),
        'dtype': torch.int32,
        'requires_grad': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Single element sparse tensor
    input_dict_7 = {
        'indices': np.array([[50], [50]], dtype=np.int64),
        'values': np.array([100.0], dtype=np.float32),
        'size': (100, 100),
        'dtype': torch.float32,
        'requires_grad': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))
    
    # Input 8: Boolean sparse tensor
    input_dict_8 = {
        'indices': np.array([[0, 1], [1, 0]], dtype=np.int64),
        'values': np.array([True, False], dtype=np.bool_),
        'size': (2, 2),
        'dtype': torch.bool,
        'requires_grad': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))
    
    # Input 9: Large size tuple
    input_dict_9 = {
        'indices': np.array([[100, 999], [200, 888]], dtype=np.int64),
        'values': np.array([10.1, 20.2], dtype=np.float32),
        'size': (1000, 1000),
        'dtype': torch.float32,
        'requires_grad': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))
    
    # Input 10: 4D sparse tensor
    input_dict_10 = {
        'indices': np.array([[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]], dtype=np.int64),
        'values': np.array([1, 2, 3, 4], dtype=np.int64),
        'size': (4, 4, 4, 4),
        'dtype': torch.int64,
        'requires_grad': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: another requires_grad = True example
    input_dict_11 = {
        'indices': np.array([[0]], dtype=np.int64),
        'values': np.array([1.0], dtype=np.float64),
        'size': (1,),
        'dtype': torch.float64,
        'requires_grad': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    return list_of_inputs

generated_inputs["torch.sparse_coo_tensor_2"] = torch_sparse_coo_tensor_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.sparse_coo_tensor_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sparse_coo_tensor_2'.")

check_valid('torch.sparse_coo_tensor', generated_inputs['torch.sparse_coo_tensor_2'], lib="torch", suffix=2)
