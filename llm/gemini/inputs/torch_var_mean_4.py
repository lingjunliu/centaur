
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def var_mean_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D tensor, all dims
    input_dict = {
        'input': np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        'unbiased': True,
        'dim': [],
        'keepdim': False,
        'out': (np.array(0., dtype=np.float32), np.array(0., dtype=np.float32))
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D tensor, all dims, biased
    input_dict = {
        'input': np.arange(12, dtype=np.float32).reshape(3, 4),
        'unbiased': False,
        'dim': [],
        'keepdim': False,
        'out': (np.array(0., dtype=np.float32), np.array(0., dtype=np.float32))
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D tensor, dim=1
    input_dict = {
        'input': np.arange(24, dtype=np.float32).reshape(2, 3, 4),
        'dim': [1],
        'unbiased': True,
        'keepdim': False,
        'out': (np.zeros((2, 4), dtype=np.float32), np.zeros((2, 4), dtype=np.float32))
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D tensor, dim=[0, 2]
    input_dict = {
        'input': np.arange(24, dtype=np.float32).reshape(2, 3, 4),
        'dim': [0, 2],
        'unbiased': False,
        'keepdim': False,
        'out': (np.zeros((3,), dtype=np.float32), np.zeros((3,), dtype=np.float32))
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D tensor, keepdim=True
    input_dict = {
        'input': np.array([[1., 2., 3.], [4., 5., 6.]], dtype=np.float32),
        'dim': [0],
        'keepdim': True,
        'unbiased': True,
        'out': (np.zeros((1, 3), dtype=np.float32), np.zeros((1, 3), dtype=np.float32))
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D tensor with negative values
    input_dict = {
        'input': np.array([[-1., -2., -3.], [-4., -5., -6.]], dtype=np.float32),
        'dim': [1],
        'keepdim': False,
        'unbiased': False,
        'out': (np.zeros((2,), dtype=np.float32), np.zeros((2,), dtype=np.float32))
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D tensor, float64, keepdim=True
    input_dict = {
        'input': np.random.randn(2, 3, 4, 5).astype(np.float64),
        'dim': [1, 3],
        'keepdim': True,
        'unbiased': True,
        'out': (np.zeros((2, 1, 4, 1), dtype=np.float64), np.zeros((2, 1, 4, 1), dtype=np.float64))
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Tensor with a single element
    input_dict = {
        'input': np.array([100.0], dtype=np.float32),
        'unbiased': True,
        'dim': [],
        'keepdim': False,
        'out': (np.array(0., dtype=np.float32), np.array(0., dtype=np.float32))
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: High-dimensional tensor
    input_dict = {
        'input': np.ones((2, 2, 2, 2, 2), dtype=np.float32),
        'dim': [0, 2, 4],
        'unbiased': True,
        'keepdim': False,
        'out': (np.zeros((2, 2), dtype=np.float32), np.zeros((2, 2), dtype=np.float32))
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Tensor with zero and non-zero values
    input_dict = {
        'input': np.array([[0., 1., -1.], [1., 0., -1.]], dtype=np.float32),
        'unbiased': False,
        'dim': [],
        'keepdim': False,
        'out': (np.array(0., dtype=np.float32), np.array(0., dtype=np.float32))
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.var_mean_4"] = var_mean_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.var_mean_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.var_mean_4'.")

check_valid('torch.var_mean', generated_inputs['torch.var_mean_4'], lib="torch", suffix=4)
