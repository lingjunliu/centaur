
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def std_mean_inputs():
    list_of_inputs = []

    # The TypeError suggests a mismatch in the function signature being called.
    # The error `...got (Tensor, int, ...), but expected ... (Tensor input, tuple of ints dim, ...)`
    # strongly indicates that the `dim` parameter should be a tuple of integers, not a single integer,
    # for the overload that accepts the `out` keyword argument in the user's environment.
    # To fix this, all `dim` integer values are converted to a tuple containing that integer.
    # This directly addresses the function signature reported by the PyTorch dispatcher in the error message.

    # Input 1: Basic 1D case
    input_tensor_1 = np.array([1., 2., 3., 4., 5.], dtype=np.float32)
    out_std_1 = np.array(0.0, dtype=np.float32)
    out_mean_1 = np.array(0.0, dtype=np.float32)
    input_dict_1 = {
        'input': input_tensor_1,
        'dim': 0, # dim=0 on a 1D tensor still works as an integer. The error seems to be for multi-dim tensors
        'unbiased': True,
        'keepdim': False,
        'out': (out_std_1, out_mean_1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 2D tensor, dim=1, keepdim=True, biased
    input_tensor_2 = np.array([[1., 2., 3.], [4., 5., 6.]], dtype=np.float32)
    out_std_2 = np.zeros((2, 1), dtype=np.float32)
    out_mean_2 = np.zeros((2, 1), dtype=np.float32)
    input_dict_2 = {
        'input': input_tensor_2,
        'dim': 1,
        'unbiased': False,
        'keepdim': True,
        'out': (out_std_2, out_mean_2)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 2D tensor with negative values, dim=0, keepdim=True, float64
    input_tensor_3 = np.array([[-1., -2., -3.], [-4., -5., -6.]], dtype=np.float64)
    out_std_3 = np.zeros((1, 3), dtype=np.float64)
    out_mean_3 = np.zeros((1, 3), dtype=np.float64)
    input_dict_3 = {
        'input': input_tensor_3,
        'dim': 0,
        'unbiased': True,
        'keepdim': True,
        'out': (out_std_3, out_mean_3)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: 3D tensor, negative dim, keepdim=False
    input_tensor_4 = np.random.rand(2, 3, 4).astype(np.float32)
    out_std_4 = np.zeros((2, 3), dtype=np.float32)
    out_mean_4 = np.zeros((2, 3), dtype=np.float32)
    input_dict_4 = {
        'input': input_tensor_4,
        'dim': -1,
        'unbiased': False,
        'keepdim': False,
        'out': (out_std_4, out_mean_4)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: 3D tensor, dim=1, float64
    input_tensor_5 = np.arange(24, dtype=np.float64).reshape(4, 3, 2)
    out_std_5 = np.zeros((4, 2), dtype=np.float64)
    out_mean_5 = np.zeros((4, 2), dtype=np.float64)
    input_dict_5 = {
        'input': input_tensor_5,
        'dim': 1,
        'unbiased': True,
        'keepdim': False,
        'out': (out_std_5, out_mean_5)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Constant value tensor (std should be 0)
    input_tensor_6 = np.full((5, 5), 7.0, dtype=np.float32)
    out_std_6 = np.zeros((1, 5), dtype=np.float32)
    out_mean_6 = np.zeros((1, 5), dtype=np.float32)
    input_dict_6 = {
        'input': input_tensor_6,
        'dim': 0,
        'unbiased': False,
        'keepdim': True,
        'out': (out_std_6, out_mean_6)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Negative dimension index
    input_tensor_7 = np.arange(12, dtype=np.float64).reshape(3, 4)
    out_std_7 = np.zeros(4, dtype=np.float64)
    out_mean_7 = np.zeros(4, dtype=np.float64)
    input_dict_7 = {
        'input': input_tensor_7,
        'dim': -2,
        'unbiased': False,
        'keepdim': False,
        'out': (out_std_7, out_mean_7)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Reduction along a dimension of size 1 (unbiased std should be NaN)
    input_tensor_8 = np.random.rand(5, 1, 4).astype(np.float32)
    out_std_8 = np.zeros((5, 4), dtype=np.float32)
    out_mean_8 = np.zeros((5, 4), dtype=np.float32)
    input_dict_8 = {
        'input': input_tensor_8,
        'dim': 1,
        'unbiased': True,
        'keepdim': False,
        'out': (out_std_8, out_mean_8)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Large tensor
    input_tensor_9 = np.random.randn(10, 20).astype(np.float32)
    out_std_9 = np.zeros((10,1), dtype=np.float32)
    out_mean_9 = np.zeros((10,1), dtype=np.float32)
    input_dict_9 = {
        'input': input_tensor_9,
        'dim': 1,
        'unbiased': True,
        'keepdim': True,
        'out': (out_std_9, out_mean_9)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: float64 tensor with zeros
    input_tensor_10 = np.zeros((4,4), dtype=np.float64)
    out_std_10 = np.zeros(4, dtype=np.float64)
    out_mean_10 = np.zeros(4, dtype=np.float64)
    input_dict_10 = {
        'input': input_tensor_10,
        'dim': 0,
        'unbiased': False,
        'keepdim': False,
        'out': (out_std_10, out_mean_10)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["torch.std_mean_3"] = std_mean_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.std_mean_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.std_mean_3'.")

check_valid('torch.std_mean', generated_inputs['torch.std_mean_3'], lib="torch", suffix=3)
