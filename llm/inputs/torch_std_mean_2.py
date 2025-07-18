
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def std_mean_inputs():
    list_of_inputs = []

    # The user is encountering a conflicting error cycle.
    # 1. A `TypeError` occurs if the 'out' parameter is provided, because their PyTorch version does not support it for this function.
    # 2. A `KeyError` occurs if the 'out' parameter is omitted, because their test harness's signature definition strictly requires it.
    # This indicates a mismatch between the test harness's signature and the actual library behavior.
    # To resolve the most recent error (`KeyError: 'out'`), the 'out' parameter must be included in the input dictionary.
    # The following inputs adhere strictly to the provided signature, including the 'out' parameter with correctly shaped tensors.

    # Input 1: Basic 1D tensor, reduce all dimensions
    input_dict_1 = {
        'input': np.array([1., 2., 3., 4., 5.], dtype=np.float32),
        'dim': [],
        'unbiased': True,
        'keepdim': False,
        'out': (np.empty((), dtype=np.float32), np.empty((), dtype=np.float32))
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 2D tensor, dim=0, unbiased=False
    input_dict_2 = {
        'input': np.array([[1., 2., 3.], [4., 5., 6.]], dtype=np.float32),
        'dim': [0],
        'unbiased': False,
        'keepdim': False,
        'out': (np.empty(3, dtype=np.float32), np.empty(3, dtype=np.float32))
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 2D tensor, dim=1, keepdim=True, float64
    input_dict_3 = {
        'input': np.array([[-1., -2., -3.], [4., 5., 6.]], dtype=np.float64),
        'dim': [1],
        'unbiased': True,
        'keepdim': True,
        'out': (np.empty((2, 1), dtype=np.float64), np.empty((2, 1), dtype=np.float64))
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: 3D tensor, multiple dims, keepdim=True, unbiased=False
    input_dict_4 = {
        'input': np.arange(24, dtype=np.float32).reshape(2, 3, 4),
        'dim': [0, 2],
        'unbiased': False,
        'keepdim': True,
        'out': (np.empty((1, 3, 1), dtype=np.float32), np.empty((1, 3, 1), dtype=np.float32))
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Tensor with negative values and zeros, reduce all dims
    input_dict_5 = {
        'input': np.array([-1., 0., -3., -4., 0.], dtype=np.float32),
        'dim': [],
        'unbiased': True,
        'keepdim': False,
        'out': (np.empty((), dtype=np.float32), np.empty((), dtype=np.float32))
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: 3D tensor, negative dim
    input_dict_6 = {
        'input': np.arange(24, dtype=np.float64).reshape(2, 3, 4) * -1,
        'dim': [-1],
        'unbiased': True,
        'keepdim': False,
        'out': (np.empty((2, 3), dtype=np.float64), np.empty((2, 3), dtype=np.float64))
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: keepdim=True and unbiased=False
    input_dict_7 = {
        'input': np.array([[10., 20.], [30., 40.], [50., 60.]], dtype=np.float32),
        'dim': [0],
        'unbiased': False,
        'keepdim': True,
        'out': (np.empty((1, 2), dtype=np.float32), np.empty((1, 2), dtype=np.float32))
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: High dimensional tensor
    input_dict_8 = {
        'input': np.arange(120, dtype=np.float64).reshape(2, 3, 4, 5),
        'dim': [1, 3],
        'unbiased': False,
        'keepdim': True,
        'out': (np.empty((2, 1, 4, 1), dtype=np.float64), np.empty((2, 1, 4, 1), dtype=np.float64))
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Tensor with a single element, unbiased=True (std is nan)
    input_dict_9 = {
        'input': np.array([42.], dtype=np.float32),
        'dim': [],
        'unbiased': True,
        'keepdim': False,
        'out': (np.empty((), dtype=np.float32), np.empty((), dtype=np.float32))
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Tensor with a single element, unbiased=False (std is 0)
    input_dict_10 = {
        'input': np.array([42.], dtype=np.float64),
        'dim': [],
        'unbiased': False,
        'keepdim': False,
        'out': (np.empty((), dtype=np.float64), np.empty((), dtype=np.float64))
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["torch.std_mean_2"] = std_mean_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.std_mean_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.std_mean_2'.")

check_valid('torch.std_mean', generated_inputs['torch.std_mean_2'], lib="torch", suffix=2)
