
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def torch_std_mean_inputs():
    """
    Generates a list of valid inputs for the torch.std_mean API.
    The test harness requires the 'out' key, and the calling order of keyword
    arguments matters for the torch.std_mean C++ dispatcher. The order
    'unbiased', 'keepdim', 'out' must be preserved.
    """
    list_of_inputs = []

    # Input 1: 2D tensor, reduce dim 0, keepdim=False
    input_tensor_1 = torch.arange(1., 7., dtype=torch.float32).reshape(2, 3).numpy()
    out_std_1 = torch.empty(3, dtype=torch.float32).numpy()
    out_mean_1 = torch.empty(3, dtype=torch.float32).numpy()
    input_dict_1 = {
        'input': input_tensor_1,
        'dim': (0,),
        'unbiased': True,
        'keepdim': False,
        'out': (out_std_1, out_mean_1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 2D tensor, reduce dim 1, keepdim=True
    input_tensor_2 = torch.tensor([[1, 5, 3], [4, 2, 6]], dtype=torch.float32).numpy()
    out_std_2 = torch.empty((2, 1), dtype=torch.float32).numpy()
    out_mean_2 = torch.empty((2, 1), dtype=torch.float32).numpy()
    input_dict_2 = {
        'input': input_tensor_2,
        'dim': (1,),
        'unbiased': False,
        'keepdim': True,
        'out': (out_std_2, out_mean_2)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 3D tensor, reduce multiple dims, keepdim=True
    input_tensor_3 = torch.randn(2, 3, 4).numpy()
    out_std_3 = torch.empty((1, 3, 1), dtype=torch.float32).numpy()
    out_mean_3 = torch.empty((1, 3, 1), dtype=torch.float32).numpy()
    input_dict_3 = {
        'input': input_tensor_3,
        'dim': (0, 2),
        'unbiased': True,
        'keepdim': True,
        'out': (out_std_3, out_mean_3)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: 3D tensor, reduce single dim, keepdim=False, float64
    input_tensor_4 = torch.randn(2, 3, 4, dtype=torch.float64).numpy()
    out_std_4 = torch.empty((2, 4), dtype=torch.float64).numpy()
    out_mean_4 = torch.empty((2, 4), dtype=torch.float64).numpy()
    input_dict_4 = {
        'input': input_tensor_4,
        'dim': (1,),
        'unbiased': False,
        'keepdim': False,
        'out': (out_std_4, out_mean_4)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Tensor with negative values, reduce all dims by specifying all of them
    input_tensor_5 = torch.tensor([[-1., -2., -3.], [1., 2., 3.]], dtype=torch.float32).numpy()
    out_std_5 = torch.empty((), dtype=torch.float32).numpy()
    out_mean_5 = torch.empty((), dtype=torch.float32).numpy()
    input_dict_5 = {
        'input': input_tensor_5,
        'dim': (0, 1),
        'unbiased': True,
        'keepdim': False,
        'out': (out_std_5, out_mean_5)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: 1D Float64 input tensor
    input_tensor_6 = torch.tensor([10.5, 20.3, 0.1, -5.2], dtype=torch.float64).numpy()
    out_std_6 = torch.empty((), dtype=torch.float64).numpy()
    out_mean_6 = torch.empty((), dtype=torch.float64).numpy()
    input_dict_6 = {
        'input': input_tensor_6,
        'dim': (0,),
        'unbiased': False,
        'keepdim': False,
        'out': (out_std_6, out_mean_6)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: 4D tensor, reduce dimensions (1, 3), keepdim=True
    input_tensor_7 = torch.rand(2, 5, 3, 4).numpy()
    out_std_7 = torch.empty((2, 1, 3, 1), dtype=torch.float32).numpy()
    out_mean_7 = torch.empty((2, 1, 3, 1), dtype=torch.float32).numpy()
    input_dict_7 = {
        'input': input_tensor_7,
        'dim': (1, 3),
        'unbiased': True,
        'keepdim': True,
        'out': (out_std_7, out_mean_7)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: 4D tensor, reduce dimensions (0, 2), keepdim=False
    input_tensor_8 = torch.rand(2, 5, 3, 4).numpy()
    out_std_8 = torch.empty((5, 4), dtype=torch.float32).numpy()
    out_mean_8 = torch.empty((5, 4), dtype=torch.float32).numpy()
    input_dict_8 = {
        'input': input_tensor_8,
        'dim': (0, 2),
        'unbiased': False,
        'keepdim': False,
        'out': (out_std_8, out_mean_8)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Complex input tensor. Std is real, mean is complex.
    input_tensor_9 = (torch.randn(4, 3, dtype=torch.cfloat) + torch.randn(4, 3, dtype=torch.cfloat) * 1j).numpy()
    out_std_9 = torch.empty(4, dtype=torch.float32).numpy()
    out_mean_9 = torch.empty(4, dtype=torch.cfloat).numpy()
    input_dict_9 = {
        'input': input_tensor_9,
        'dim': (1,),
        'unbiased': True,
        'keepdim': False,
        'out': (out_std_9, out_mean_9)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Tensor with a dimension of size 1 (std should be 0)
    input_tensor_10 = torch.tensor([[1.], [2.], [3.], [4.]], dtype=torch.float32).numpy()
    out_std_10 = torch.empty(4, dtype=torch.float32).numpy()
    out_mean_10 = torch.empty(4, dtype=torch.float32).numpy()
    input_dict_10 = {
        'input': input_tensor_10,
        'dim': (1,),
        'unbiased': False,
        'keepdim': False,
        'out': (out_std_10, out_mean_10)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["torch.std_mean_4"] = torch_std_mean_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.std_mean_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.std_mean_4'.")

check_valid('torch.std_mean', generated_inputs['torch.std_mean_4'], lib="torch", suffix=4)
