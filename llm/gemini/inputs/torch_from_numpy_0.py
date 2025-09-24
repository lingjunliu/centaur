
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import copy
import numpy as np

def from_numpy_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array
    input_dict_1 = {
        'ndarray': np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 2D int64 array
    input_dict_2 = {
        'ndarray': np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 3D uint8 array
    input_dict_3 = {
        'ndarray': np.arange(8, dtype=np.uint8).reshape(2, 2, 2)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: 1D boolean array
    input_dict_4 = {
        'ndarray': np.array([True, False, True, False], dtype=np.bool_)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: 0D array (scalar)
    input_dict_5 = {
        'ndarray': np.array(42, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Array with a single element
    input_dict_6 = {
        'ndarray': np.array([-99.9], dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))
    
    # Input 7: 2D float64 array
    input_dict_7 = {
        'ndarray': np.random.rand(4, 2).astype(np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: 1D int16 array
    input_dict_8 = {
        'ndarray': np.array([-100, 0, 100, 200, 300], dtype=np.int16)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Contiguous array
    input_dict_9 = {
        'ndarray': np.ascontiguousarray(np.arange(10, dtype=np.int32).reshape(2, 5).T)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))
    
    # Input 10: Another contiguous array
    contiguous_array_10 = np.arange(25, dtype=np.float32).reshape(5, 5)
    input_dict_10 = {
        'ndarray': np.ascontiguousarray(contiguous_array_10[::2, 1::2])
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["torch.from_numpy"] = from_numpy_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.from_numpy' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.from_numpy'.")

check_valid('torch.from_numpy', generated_inputs['torch.from_numpy'], lib="torch", suffix=0)
