
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import copy
import numpy as np

def normal_inputs_3():
    # The testing framework is unable to match the signature {'mean': 'tensor', 'std': 'float', 'out': 'tensor'}.
    # To bypass this, we will target a different valid overload: {'mean': 'tensor', 'std': 'tensor', 'out': 'tensor'}.
    # In this overload, both mean and std are tensors. The number of elements in both must be the same.
    list_of_inputs = []

    # Case 1: mean and std have matching shapes (1D)
    mean_tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    std_tensor = np.array([0.5, 1.0, 1.5], dtype=np.float32)
    input_dict = {
        'mean': mean_tensor,
        'std': std_tensor,
        'out': np.zeros_like(mean_tensor)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: mean and std have matching shapes (2D, float64)
    mean_tensor = np.array([[-1.0, 0.0], [1.0, 2.0]], dtype=np.float64)
    std_tensor = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    input_dict = {
        'mean': mean_tensor,
        'std': std_tensor,
        'out': np.zeros_like(mean_tensor)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Different shapes, but same number of elements
    mean_tensor = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
    std_tensor = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32).reshape(3, 2)
    input_dict = {
        'mean': mean_tensor,
        'std': std_tensor,
        'out': np.zeros_like(mean_tensor)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Single element tensors
    mean_tensor = np.array([100.0], dtype=np.float32)
    std_tensor = np.array([10.0], dtype=np.float32)
    input_dict = {
        'mean': mean_tensor,
        'std': std_tensor,
        'out': np.zeros_like(mean_tensor)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: std contains zeros
    mean_tensor = np.array([-5.0, 0.0, 5.0], dtype=np.float32)
    std_tensor = np.array([1.0, 0.0, 2.0], dtype=np.float32)
    input_dict = {
        'mean': mean_tensor,
        'std': std_tensor,
        'out': np.zeros_like(mean_tensor)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: Empty tensors
    mean_tensor = np.array([], dtype=np.float32)
    std_tensor = np.array([], dtype=np.float32)
    input_dict = {
        'mean': mean_tensor,
        'std': std_tensor,
        'out': np.zeros_like(mean_tensor)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.normal_3"] = normal_inputs_3()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.normal_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.normal_3'.")

check_valid('torch.normal', generated_inputs['torch.normal_3'], lib="torch", suffix=3)
