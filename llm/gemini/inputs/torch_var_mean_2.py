
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def torch_var_mean_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D tensor
    input_dict_1 = {
        'input': np.array([1., 2., 3., 4., 5.], dtype=np.float32),
        'dim': [0],
        'unbiased': True,
        'keepdim': False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 2D tensor, reduce all dimensions, biased
    input_dict_2 = {
        'input': np.array([[1., 2., 3.], [4., 5., 6.]], dtype=np.float32),
        'dim': [0, 1],
        'unbiased': False,
        'keepdim': False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 2D tensor, reduce one dimension
    input_dict_3 = {
        'input': np.array([[1., 2., 3.], [4., 5., 6.]], dtype=np.float32),
        'dim': [0],
        'unbiased': True,
        'keepdim': False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: 2D tensor, reduce one dimension and keepdim
    input_dict_4 = {
        'input': np.array([[1., 2., 3.], [4., 5., 6.]], dtype=np.float32),
        'dim': [1],
        'unbiased': True,
        'keepdim': True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: 3D tensor with negative values and negative dim index
    input_dict_5 = {
        'input': np.array([[[-1., -2.], [-3., -4.]], [[1., 2.], [3., 4.]]], dtype=np.float32),
        'dim': [-1],
        'unbiased': False,
        'keepdim': True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: 3D tensor, reduce multiple dimensions
    input_dict_6 = {
        'input': np.random.rand(2, 3, 4).astype(np.float32),
        'dim': [0, 2],
        'unbiased': False,
        'keepdim': False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Float64 tensor
    input_dict_7 = {
        'input': np.array([10.5, 20.5, 30.5, 40.5, 50.5], dtype=np.float64),
        'dim': [0],
        'unbiased': True,
        'keepdim': False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Single element tensor, unbiased (variance is NaN)
    input_dict_8 = {
        'input': np.array([100.], dtype=np.float32),
        'dim': [0],
        'unbiased': True,
        'keepdim': False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Single element tensor, biased (variance is 0)
    input_dict_9 = {
        'input': np.array([100.], dtype=np.float32),
        'dim': [0],
        'unbiased': False,
        'keepdim': False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))
    
    # Input 10: 4D tensor with multi-dim reduction
    input_dict_10 = {
        'input': np.random.randn(2, 3, 4, 5).astype(np.float32),
        'dim': [1, 3],
        'unbiased': False,
        'keepdim': True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["torch.var_mean_2"] = torch_var_mean_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.var_mean_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.var_mean_2'.")

check_valid('torch.var_mean', generated_inputs['torch.var_mean_2'], lib="torch", suffix=2)
