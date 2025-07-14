
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def normal_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D tensor
    tensor = np.zeros(5, dtype=np.float32)
    mean = 0.0
    std = 1.0
    input_dict = {"tensor": tensor, "mean": mean, "std": std}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D tensor
    tensor = np.zeros((3, 4), dtype=np.float64)
    mean = 0.5
    std = 0.2
    input_dict = {"tensor": tensor, "mean": mean, "std": std}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D tensor
    tensor = np.zeros((2, 3, 2), dtype=np.float32)
    mean = -0.1
    std = 0.5
    input_dict = {"tensor": tensor, "mean": mean, "std": std}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Large standard deviation
    tensor = np.zeros((4, 4), dtype=np.float64)
    mean = 0.0
    std = 5.0
    input_dict = {"tensor": tensor, "mean": mean, "std": std}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Non-zero mean
    tensor = np.zeros((2, 2, 2), dtype=np.float32)
    mean = 2.0
    std = 1.0
    input_dict = {"tensor": tensor, "mean": mean, "std": std}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Small standard deviation
    tensor = np.zeros(10, dtype=np.float64)
    mean = 0.0
    std = 0.01
    input_dict = {"tensor": tensor, "mean": mean, "std": std}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Negative mean
    tensor = np.zeros((5, 5), dtype=np.float32)
    mean = -1.0
    std = 0.5
    input_dict = {"tensor": tensor, "mean": mean, "std": std}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Different data type
    tensor = np.zeros(7, dtype=np.float64)
    mean = 0.0
    std = 1.0
    input_dict = {"tensor": tensor, "mean": mean, "std": std}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Larger tensor
    tensor = np.zeros((10, 10), dtype=np.float32)
    mean = 0.2
    std = 0.3
    input_dict = {"tensor": tensor, "mean": mean, "std": std}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D tensor
    tensor = np.zeros((2, 2, 2, 2), dtype=np.float64)
    mean = -0.5
    std = 0.75
    input_dict = {"tensor": tensor, "mean": mean, "std": std}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.init.normal_"] = normal_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.init.normal_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.init.normal_'.")

check_valid('torch.nn.init.normal_', generated_inputs['torch.nn.init.normal_'], lib="torch", suffix=0)
