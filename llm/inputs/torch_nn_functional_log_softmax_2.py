
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy
import copy

def log_softmax_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D tensor
    input_dict_1 = {
        'input': numpy.array([1.0, 2.0, 3.0], dtype=numpy.float32),
        'dim': 0,
        'dtype': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 2D tensor, dim=1
    input_dict_2 = {
        'input': numpy.array([[1.0, 2.0], [3.0, 4.0]], dtype=numpy.float32),
        'dim': 1,
        'dtype': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 2D tensor, dim=0
    input_dict_3 = {
        'input': numpy.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=numpy.float32),
        'dim': 0,
        'dtype': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: 2D tensor with negative values, dim=-1
    input_dict_4 = {
        'input': numpy.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=numpy.float32),
        'dim': -1,
        'dtype': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: 3D tensor, dim=2
    input_dict_5 = {
        'input': numpy.random.rand(2, 3, 4).astype(numpy.float32),
        'dim': 2,
        'dtype': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: 3D tensor with mixed values, dim=-2
    input_dict_6 = {
        'input': numpy.array([[[1., -1.], [0., 2.]], [[-2., 3.], [4., -4.]]], dtype=numpy.float32),
        'dim': -2,
        'dtype': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: 4D tensor, dim=3, with specified dtype
    input_dict_7 = {
        'input': numpy.random.randn(2, 3, 4, 5).astype(numpy.float32),
        'dim': 3,
        'dtype': torch.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Large 2D tensor
    input_dict_8 = {
        'input': numpy.arange(20, dtype=numpy.float32).reshape(4, 5),
        'dim': 1,
        'dtype': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Float64 tensor
    input_dict_9 = {
        'input': numpy.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=numpy.float64),
        'dim': 0,
        'dtype': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: 1D tensor with a single element
    input_dict_10 = {
        'input': numpy.array([100.0], dtype=numpy.float32),
        'dim': 0,
        'dtype': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: Tensor with all zeros
    input_dict_11 = {
        'input': numpy.zeros((3, 3), dtype=numpy.float32),
        'dim': 1,
        'dtype': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    # Input 12: High-dimensional tensor, dim=-1, with specified dtype
    input_dict_12 = {
        'input': numpy.random.rand(1, 2, 3, 4, 5).astype(numpy.float64),
        'dim': -1,
        'dtype': torch.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict_12))

    return list_of_inputs

generated_inputs["torch.nn.functional.log_softmax_2"] = log_softmax_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.log_softmax_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.log_softmax_2'.")

check_valid('torch.nn.functional.log_softmax', generated_inputs['torch.nn.functional.log_softmax_2'], lib="torch", suffix=2)
