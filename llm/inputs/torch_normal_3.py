
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def normal_inputs():
    list_of_inputs = []

    # Input 1
    mean = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    std = np.float32(1.0)
    out = np.zeros_like(mean, dtype=np.float32)

    input_dict = {
        "mean": mean,
        "std": std,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    mean = np.array([1.0, 2.0], dtype=np.float32)
    std = np.float32(0.5)
    out = np.zeros_like(mean, dtype=np.float32)
    input_dict = {
        "mean": mean,
        "std": std,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    mean = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    std = np.float32(2.0)
    out = np.zeros_like(mean, dtype=np.float32)

    input_dict = {
        "mean": mean,
        "std": std,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    mean = np.array([0.0], dtype=np.float32)
    std = np.float32(0.1)
    out = np.zeros_like(mean, dtype=np.float32)
    input_dict = {
        "mean": mean,
        "std": std,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    mean = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    std = np.float32(1.5)
    out = np.zeros_like(mean, dtype=np.float32)

    input_dict = {
        "mean": mean,
        "std": std,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    mean = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    std = np.float32(0.75)
    out = np.zeros_like(mean, dtype=np.float32)

    input_dict = {
        "mean": mean,
        "std": std,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    mean = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    std = np.float32(0.25)
    out = np.zeros_like(mean, dtype=np.float32)

    input_dict = {
        "mean": mean,
        "std": std,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    mean = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    std = np.float32(0.8)
    out = np.zeros_like(mean, dtype=np.float32)

    input_dict = {
        "mean": mean,
        "std": std,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    mean = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float32)
    std = np.float32(1.2)
    out = np.zeros_like(mean, dtype=np.float32)

    input_dict = {
        "mean": mean,
        "std": std,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    mean = np.array([[-1.0, 1.0], [-2.0, 2.0]], dtype=np.float32)
    std = np.float32(0.9)
    out = np.zeros_like(mean, dtype=np.float32)

    input_dict = {
        "mean": mean,
        "std": std,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.normal_3"] = normal_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.normal_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.normal_3'.")

check_valid('torch.normal', generated_inputs['torch.normal_3'], lib="torch", suffix=3)
