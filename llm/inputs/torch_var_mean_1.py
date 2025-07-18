
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import copy
import numpy as np

def torch_var_mean_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D float tensor
    input_tensor = np.array([1., 2., 3., 4., 5.], dtype=np.float32)
    out_var = np.empty((), dtype=np.float32)
    out_mean = np.empty((), dtype=np.float32)
    input_dict = {
        'input': input_tensor,
        'dim': [0],
        'unbiased': True,
        'keepdim': False,
        'out': (out_var, out_mean)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D tensor, biased variance, keepdim=True
    input_tensor = np.array([[1., 2., 3.], [4., 5., 6.]], dtype=np.float32)
    out_var = np.empty((2, 1), dtype=np.float32)
    out_mean = np.empty((2, 1), dtype=np.float32)
    input_dict = {
        'input': input_tensor,
        'dim': [1],
        'unbiased': False,
        'keepdim': True,
        'out': (out_var, out_mean)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D tensor with negative values, multiple dims, float64
    input_tensor = np.array([[-1., 0., 5.], [-10., 2.5, -3.1]], dtype=np.float64)
    out_var = np.empty((), dtype=np.float64)
    out_mean = np.empty((), dtype=np.float64)
    input_dict = {
        'input': input_tensor,
        'dim': [0, 1],
        'unbiased': True,
        'keepdim': False,
        'out': (out_var, out_mean)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D tensor, negative dim index, keepdim=True
    input_tensor = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    out_var = np.empty((2, 3, 1), dtype=np.float32)
    out_mean = np.empty((2, 3, 1), dtype=np.float32)
    input_dict = {
        'input': input_tensor,
        'dim': [-1],
        'unbiased': False,
        'keepdim': True,
        'out': (out_var, out_mean)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 3D tensor, multiple dims, keepdim=False
    input_tensor = np.random.randn(2, 3, 4).astype(np.float32)
    out_var = np.empty((3,), dtype=np.float32)
    out_mean = np.empty((3,), dtype=np.float32)
    input_dict = {
        'input': input_tensor,
        'dim': [0, 2],
        'unbiased': True,
        'keepdim': False,
        'out': (out_var, out_mean)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Tensor with a single element (biased to avoid NaN)
    input_tensor = np.array([10.0], dtype=np.float32)
    out_var = np.empty((), dtype=np.float32)
    out_mean = np.empty((), dtype=np.float32)
    input_dict = {
        'input': input_tensor,
        'dim': [0],
        'unbiased': False,
        'keepdim': False,
        'out': (out_var, out_mean)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Tensor where all elements are the same (variance should be 0)
    input_tensor = np.full((4, 4), 7.7, dtype=np.float64)
    out_var = np.empty((), dtype=np.float64)
    out_mean = np.empty((), dtype=np.float64)
    input_dict = {
        'input': input_tensor,
        'dim': [0, 1],
        'unbiased': True,
        'keepdim': False,
        'out': (out_var, out_mean)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D tensor, reducing multiple dimensions
    input_tensor = np.random.rand(2, 3, 4, 5).astype(np.float32)
    out_var = np.empty((2, 1, 1, 5), dtype=np.float32)
    out_mean = np.empty((2, 1, 1, 5), dtype=np.float32)
    input_dict = {
        'input': input_tensor,
        'dim': [1, 2],
        'unbiased': False,
        'keepdim': True,
        'out': (out_var, out_mean)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Larger 1D tensor
    input_tensor = np.linspace(-50, 50, 100, dtype=np.float64)
    out_var = np.empty((1,), dtype=np.float64)
    out_mean = np.empty((1,), dtype=np.float64)
    input_dict = {
        'input': input_tensor,
        'dim': [0],
        'unbiased': True,
        'keepdim': True,
        'out': (out_var, out_mean)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: All dimensions reduced
    input_tensor = np.random.randn(3, 4, 5).astype(np.float32)
    out_var = np.empty((), dtype=np.float32)
    out_mean = np.empty((), dtype=np.float32)
    input_dict = {
        'input': input_tensor,
        'dim': [0, 1, 2],
        'unbiased': True,
        'keepdim': False,
        'out': (out_var, out_mean)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.var_mean_1"] = torch_var_mean_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.var_mean_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.var_mean_1'.")

check_valid('torch.var_mean', generated_inputs['torch.var_mean_1'], lib="torch", suffix=1)
