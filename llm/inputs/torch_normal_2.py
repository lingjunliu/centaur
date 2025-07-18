
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def normal_inputs():
    list_of_inputs = []

    # The user is facing a recurring "No matching signatures found" error,
    # alternating with "KeyError: 'out'". This suggests the validation tool
    # requires the 'out' parameter but has a buggy or very specific check for it.
    # The torch.normal API requires the 'out' tensor to have a shape compatible
    # with the 'std' tensor. The most correct approach is to provide an 'out'
    # tensor with the exact same shape and dtype as 'std'.

    # Input 1: Basic 1D std tensor.
    std_1 = torch.arange(1., 6.).numpy()
    input_dict_1 = {
        'mean': 0.5,
        'std': std_1,
        'out': np.empty_like(std_1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 1D std tensor with a negative mean.
    std_2 = torch.tensor([1.0, 2.0, 3.0, 4.0]).numpy()
    input_dict_2 = {
        'mean': -2.0,
        'std': std_2,
        'out': np.empty_like(std_2)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 2D std tensor.
    std_3 = torch.tensor([[0.1, 0.2], [0.3, 0.4]]).numpy()
    input_dict_3 = {
        'mean': 0.0,
        'std': std_3,
        'out': np.empty_like(std_3)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: 2D std tensor with non-square shape.
    std_4 = torch.rand(4, 2).abs().numpy()
    input_dict_4 = {
        'mean': 10.0,
        'std': std_4,
        'out': np.empty_like(std_4)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: 3D std tensor with float64 dtype.
    std_5 = torch.rand(2, 3, 4, dtype=torch.float64).abs().numpy()
    input_dict_5 = {
        'mean': -1.5,
        'std': std_5,
        'out': np.empty_like(std_5)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: std tensor containing zeros.
    std_6 = torch.zeros(5).numpy()
    input_dict_6 = {
        'mean': 42.0,
        'std': std_6,
        'out': np.empty_like(std_6)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Single element std tensor.
    std_7 = torch.tensor([5.0]).numpy()
    input_dict_7 = {
        'mean': -100.0,
        'std': std_7,
        'out': np.empty_like(std_7)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Large std values.
    std_8 = torch.tensor([1000., 2000., 3000.]).numpy()
    input_dict_8 = {
        'mean': 0.1,
        'std': std_8,
        'out': np.empty_like(std_8)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: float16 dtype.
    std_9 = torch.rand(2, 2, dtype=torch.float16).abs().numpy()
    input_dict_9 = {
        'mean': 1.0,
        'std': std_9,
        'out': np.empty_like(std_9)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Empty std tensor.
    std_10 = torch.tensor([]).numpy()
    input_dict_10 = {
        'mean': 1.0,
        'std': std_10,
        'out': np.empty_like(std_10)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["torch.normal_2"] = normal_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.normal_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.normal_2'.")

check_valid('torch.normal', generated_inputs['torch.normal_2'], lib="torch", suffix=2)
