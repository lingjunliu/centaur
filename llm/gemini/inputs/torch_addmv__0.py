
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def addmv__inputs():
    list_of_inputs = []

    # Case 1: Basic functionality with positive floats
    input_dict_1 = {
        'input': np.array([1.0, 2.0, 3.0], dtype=np.float32),
        'mat': np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
        'vec': np.array([1.0, 2.0], dtype=np.float32),
        'beta': 1.0,
        'alpha': 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Case 2: Using float64 and different shapes
    input_dict_2 = {
        'input': np.array([10.0, 20.0], dtype=np.float64),
        'mat': np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], dtype=np.float64),
        'vec': np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        'beta': 2.0,
        'alpha': 0.5
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Case 3: Negative values in tensors and multipliers
    input_dict_3 = {
        'input': np.array([-1.0, -2.0, -3.0, -4.0], dtype=np.float32),
        'mat': np.array([[-1.0, -2.0], [-3.0, -4.0], [-5.0, -6.0], [-7.0, -8.0]], dtype=np.float32),
        'vec': np.array([-0.5, -1.5], dtype=np.float32),
        'beta': -1.0,
        'alpha': -2.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Case 4: Zero beta, input tensor is ignored
    input_dict_4 = {
        'input': np.array([100.0, 200.0], dtype=np.float32),
        'mat': np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        'vec': np.array([1.0, -1.0, 1.0], dtype=np.float32),
        'beta': 0.0,
        'alpha': 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Case 5: Zero alpha, mat-vec product is ignored (input is only scaled by beta)
    input_dict_5 = {
        'input': np.array([5.0, 10.0, 15.0], dtype=np.float32),
        'mat': np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
        'vec': np.array([10.0, 20.0], dtype=np.float32),
        'beta': 1.5,
        'alpha': 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Case 6: Both alpha and beta are zero, result should be zero tensor
    input_dict_6 = {
        'input': np.array([1.0, 1.0, 1.0], dtype=np.float32),
        'mat': np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]], dtype=np.float32),
        'vec': np.array([1.0, 1.0], dtype=np.float32),
        'beta': 0.0,
        'alpha': 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Case 7: Tall matrix (more rows than columns)
    input_dict_7 = {
        'input': np.arange(1, 6, dtype=np.float32),
        'mat': np.arange(1, 11, dtype=np.float32).reshape(5, 2),
        'vec': np.array([0.1, 0.2], dtype=np.float32),
        'beta': 0.5,
        'alpha': 2.5
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Case 8: Matrix with one column (mat (n,1), vec (1,))
    input_dict_8 = {
        'input': np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        'mat': np.array([[10.0], [20.0], [30.0], [40.0]], dtype=np.float32),
        'vec': np.array([2.0], dtype=np.float32),
        'beta': 1.0,
        'alpha': 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Case 9: Matrix with one row (mat (1,m), vec (m,), input (1,))
    input_dict_9 = {
        'input': np.array([100.0], dtype=np.float64),
        'mat': np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float64),
        'vec': np.array([-1.0, -2.0, -3.0, -4.0, -5.0], dtype=np.float64),
        'beta': 1.0,
        'alpha': -1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Case 10: Zero-valued tensors
    input_dict_10 = {
        'input': np.zeros(3, dtype=np.float32),
        'mat': np.zeros((3, 4), dtype=np.float32),
        'vec': np.zeros(4, dtype=np.float32),
        'beta': 100.0,
        'alpha': 100.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Case 11: Large magnitude values
    input_dict_11 = {
        'input': np.array([1e6, 2e6], dtype=np.float32),
        'mat': np.array([[1e5, 1e4], [1e3, 1e2]], dtype=np.float32),
        'vec': np.array([10.0, 20.0], dtype=np.float32),
        'beta': 0.1,
        'alpha': 0.2
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))
    
    return list_of_inputs

generated_inputs["torch.addmv_"] = addmv__inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.addmv_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.addmv_'.")

check_valid('torch.addmv_', generated_inputs['torch.addmv_'], lib="torch", suffix=0)
