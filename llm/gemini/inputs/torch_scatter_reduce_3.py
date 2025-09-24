
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def scatter_reduce_inputs():
    list_of_inputs = []

    # Case 1: include_self=True, no output_size. Output size is inferred from input.
    input_dict_1 = {
        'input': np.ones(10, dtype=np.float32),
        'dim': 0,
        'index': np.array([0, 2, 4, 6, 8], dtype=np.int64),
        'src': np.array([1, 2, 3, 4, 5], dtype=np.float32),
        'reduce': 'sum',
        'include_self': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Case 2: include_self=False, output_size is specified.
    input_dict_2 = {
        'input': np.empty((0,0), dtype=np.float64),
        'dim': 1,
        'index': np.array([[0, 0], [1, 1], [0, 1]], dtype=np.int64),
        'src': np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64),
        'reduce': 'mean',
        'output_size': (3, 2),
        'include_self': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Case 3: include_self=True, 2D prod.
    input_dict_3 = {
        'input': np.full((3, 5), 2.0, dtype=np.float32),
        'dim': 0,
        'index': np.array([[0, 1, 2, 0, 1], [2, 0, 0, 1, 2]], dtype=np.int64),
        'src': np.arange(1, 11, dtype=np.float32).reshape(2, 5),
        'reduce': 'prod',
        'include_self': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Case 4: include_self=False, 'amax'
    input_dict_4 = {
        'input': np.empty((0,0), dtype=np.float32),
        'dim': 0,
        'index': np.array([[0, 1, 2, 0, 1]], dtype=np.int64),
        'src': np.arange(1, 6, dtype=np.float32).reshape(1, 5),
        'reduce': 'amax',
        'output_size': (3, 5),
        'include_self': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Case 5: include_self=True, negative dim, 'amin'
    input_dict_5 = {
        'input': np.full((3, 4), 100, dtype=np.float32),
        'dim': -1,
        'index': np.array([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=np.int64),
        'src': np.array([[-10, 20, -5, 8], [-1, -9, 15, -12]], dtype=np.float32),
        'reduce': 'amin',
        'include_self': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Case 6: include_self=False, larger output_size, int32
    input_dict_6 = {
        'input': np.empty((0,), dtype=np.int32),
        'dim': 0,
        'index': np.array([0, 9, 2, 7, 4], dtype=np.int64),
        'src': np.array([10, 20, 30, 40, 50], dtype=np.int32),
        'reduce': 'sum',
        'output_size': (10,),
        'include_self': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Case 7: include_self=True, broadcasting src
    input_dict_7 = {
        'input': np.ones((4, 5), dtype=np.float32),
        'dim': 1,
        'index': np.array([[0, 1, 2, 3], [4, 3, 2, 1], [0, 0, 4, 4]], dtype=np.int64),
        'src': np.array([[10], [20], [30]], dtype=np.float32),
        'reduce': 'sum',
        'include_self': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Case 8: include_self=False, 3D tensors
    input_dict_8 = {
        'input': np.empty((0,0,0), dtype=np.float32),
        'dim': 1,
        'index': np.array([[[0, 1, 2, 0]], [[2, 1, 0, 1]]], dtype=np.int64),
        'src': np.arange(1, 9, dtype=np.float32).reshape(2, 1, 4),
        'reduce': 'prod',
        'output_size': (2, 3, 4),
        'include_self': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Case 9: include_self=True, int64
    input_dict_9 = {
        'input': np.array([1, 2, 3, 4], dtype=np.int64),
        'dim': 0,
        'index': np.array([3, 0, 2, 1, 1, 2, 0], dtype=np.int64),
        'src': np.array([10, 20, 30, 40, 50, 60, 70], dtype=np.int64),
        'reduce': 'sum',
        'include_self': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Case 10: include_self=True, empty index and src
    input_dict_10 = {
        'input': np.arange(1, 7, dtype=np.float32).reshape(2, 3),
        'dim': 0,
        'index': np.empty((0, 3), dtype=np.int64),
        'src': np.empty((0, 3), dtype=np.float32),
        'reduce': 'sum',
        'include_self': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["torch.scatter_reduce_3"] = scatter_reduce_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.scatter_reduce_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.scatter_reduce_3'.")

check_valid('torch.scatter_reduce', generated_inputs['torch.scatter_reduce_3'], lib="torch", suffix=3)
