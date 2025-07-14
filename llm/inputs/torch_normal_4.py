
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def torch_normal_inputs():
    list_of_inputs = []

    # Input 1
    mean = 2.0
    std = 3.0
    size = (1, 4)
    out = torch.empty(size, dtype=torch.float64).numpy()
    input_dict = {"mean": mean, "std": std, "size": size, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    mean = -1.0
    std = 0.5
    size = (2, 2)
    out = torch.empty(size, dtype=torch.float64).numpy()
    input_dict = {"mean": mean, "std": std, "size": size, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    mean = 0.0
    std = 1.0
    size = (5,)
    out = torch.empty(size, dtype=torch.float64).numpy()
    input_dict = {"mean": mean, "std": std, "size": size, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    mean = 5.0
    std = 2.0
    size = (2, 3, 4)
    out = torch.empty(size, dtype=torch.float64).numpy()
    input_dict = {"mean": mean, "std": std, "size": size, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    mean = -3.0
    std = 1.5
    size = (1, 1, 1)
    out = torch.empty(size, dtype=torch.float64).numpy()
    input_dict = {"mean": mean, "std": std, "size": size, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    mean = 10.0
    std = 5.0
    size = (4, 1)
    out = torch.empty(size, dtype=torch.float64).numpy()
    input_dict = {"mean": mean, "std": std, "size": size, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    mean = -5.5
    std = 2.5
    size = (10,)
    out = torch.empty(size, dtype=torch.float64).numpy()
    input_dict = {"mean": mean, "std": std, "size": size, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    mean = 1.2
    std = 0.8
    size = (3, 2, 1)
    out = torch.empty(size, dtype=torch.float64).numpy()
    input_dict = {"mean": mean, "std": std, "size": size, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    mean = -0.7
    std = 0.3
    size = (1, 5, 2)
    out = torch.empty(size, dtype=torch.float64).numpy()
    input_dict = {"mean": mean, "std": std, "size": size, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    mean = 7.0
    std = 4.0
    size = (2, 4)
    out = torch.empty(size, dtype=torch.float64).numpy()
    input_dict = {"mean": mean, "std": std, "size": size, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.normal_4"] = torch_normal_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.normal_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.normal_4'.")

check_valid('torch.normal', generated_inputs['torch.normal_4'], lib="torch", suffix=4)
