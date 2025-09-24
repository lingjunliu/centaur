
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def celu_inputs():
    list_of_inputs = []

    # Input 1: Basic case with positive, negative, and zero values
    input_dict_1 = {
        'input': torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float32).numpy(),
        'alpha': 1.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 2D tensor
    input_dict_2 = {
        'input': torch.randn(3, 4, dtype=torch.float32).numpy(),
        'alpha': 1.5,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 3D tensor with a different alpha
    input_dict_3 = {
        'input': torch.arange(-12, 12, dtype=torch.float32).reshape(2, 3, 4).numpy(),
        'alpha': 0.5,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Tensor with all positive values
    input_dict_4 = {
        'input': torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32).numpy(),
        'alpha': 2.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Tensor with all negative values
    input_dict_5 = {
        'input': torch.tensor([-0.1, -0.5, -1.2, -5.0], dtype=torch.float32).numpy(),
        'alpha': 1.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Tensor with a negative alpha
    input_dict_6 = {
        'input': torch.tensor([-3.0, -1.5, 0.0, 1.5, 3.0], dtype=torch.float32).numpy(),
        'alpha': -0.5,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Tensor with float64 dtype
    input_dict_7 = {
        'input': torch.randn(5, dtype=torch.float64).numpy(),
        'alpha': 1.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Empty tensor
    input_dict_8 = {
        'input': torch.tensor([], dtype=torch.float32).numpy(),
        'alpha': 1.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))
    
    # Input 9: High-dimensional tensor (4D)
    input_dict_9 = {
        'input': torch.randn(2, 2, 3, 2, dtype=torch.float32).numpy(),
        'alpha': 3.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Tensor with a single element
    input_dict_10 = {
        'input': torch.tensor([-5.0], dtype=torch.float32).numpy(),
        'alpha': 1.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: Tensor with all zeros
    input_dict_11 = {
        'input': torch.zeros(3, 3, dtype=torch.float32).numpy(),
        'alpha': 1.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    # Input 12: Alpha is zero
    input_dict_12 = {
        'input': torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float32).numpy(),
        'alpha': 0.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_12))


    return list_of_inputs

generated_inputs["torch.celu"] = celu_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.celu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.celu'.")

check_valid('torch.celu', generated_inputs['torch.celu'], lib="torch", suffix=0)
