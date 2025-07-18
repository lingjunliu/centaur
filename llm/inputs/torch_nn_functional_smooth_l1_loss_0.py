
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def smooth_l1_loss_inputs():
    list_of_inputs = []

    # The error "TypeError: smooth_l1_loss() got an unexpected keyword argument 'delta'"
    # indicates the function in the execution environment does not accept 'delta'.
    # This is true for PyTorch versions < 1.9 where the functional form only took
    # input, target, and reduction. We will generate inputs for that signature
    # to resolve the TypeError.

    # Input 1: Basic 1D case with 'mean' reduction
    input_dict_1 = {
        'input': np.array([0.5, 1.5, -0.5], dtype=np.float32),
        'target': np.array([0.2, 1.8, -0.7], dtype=np.float32),
        'reduction': 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Basic 1D case with 'sum' reduction
    input_dict_2 = {
        'input': np.array([1.1, 2.2, 3.3], dtype=np.float32),
        'target': np.array([1.0, 2.5, 3.0], dtype=np.float32),
        'reduction': 'sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 2D case with 'none' reduction
    input_dict_3 = {
        'input': np.array([[1, 2], [3, 4]], dtype=np.float32),
        'target': np.array([[1.5, 2.5], [2.5, 3.5]], dtype=np.float32),
        'reduction': 'none'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: 3D tensors
    input_dict_4 = {
        'input': np.random.rand(2, 3, 4).astype(np.float32),
        'target': np.random.rand(2, 3, 4).astype(np.float32),
        'reduction': 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Input and target with negative values
    input_dict_5 = {
        'input': np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32),
        'target': np.array([[-1.1, -2.2], [-3.3, -4.4]], dtype=np.float32),
        'reduction': 'sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Using float64 dtype
    input_dict_6 = {
        'input': np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64),
        'target': np.array([[0.11, 0.22], [0.33, 0.44]], dtype=np.float64),
        'reduction': 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))
    
    # Input 7: Empty tensors
    input_dict_7 = {
        'input': np.array([], dtype=np.float32),
        'target': np.array([], dtype=np.float32),
        'reduction': 'sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Large values in tensors
    input_dict_8 = {
        'input': np.array([1e5, 2e5, 3e5], dtype=np.float32),
        'target': np.array([1e5 + 1, 2e5 - 1, 3e5], dtype=np.float32),
        'reduction': 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Mixed positive and negative values
    input_dict_9 = {
        'input': np.array([-0.5, 0.5, -1.5, 1.5], dtype=np.float32),
        'target': np.array([-0.6, 0.4, -1.7, 1.3], dtype=np.float32),
        'reduction': 'none'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))
    
    # Input 10: Tensors with zeros
    input_dict_10 = {
        'input': np.zeros((2, 2), dtype=np.float32),
        'target': np.ones((2, 2), dtype=np.float32),
        'reduction': 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["torch.nn.functional.smooth_l1_loss"] = smooth_l1_loss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.smooth_l1_loss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.smooth_l1_loss'.")

check_valid('torch.nn.functional.smooth_l1_loss', generated_inputs['torch.nn.functional.smooth_l1_loss'], lib="torch", suffix=0)
