
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def xavier_uniform_inputs():
    list_of_inputs = []

    # Input 1
    tensor = np.zeros((3, 5))
    gain = 1.0
    input_dict = {"tensor": tensor, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    tensor = np.ones((2, 2, 2))
    gain = 0.5
    input_dict = {"tensor": tensor, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    tensor = np.random.rand(4, 4).astype(np.float32)
    gain = 2.0
    input_dict = {"tensor": tensor, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    tensor = np.random.randn(2, 3, 4).astype(np.float64)
    gain = 1.7
    input_dict = {"tensor": tensor, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    tensor = np.eye(5)
    gain = 0.3
    input_dict = {"tensor": tensor, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    tensor = np.random.rand(2, 2, 2, 2).astype(np.float32)
    gain = 0.9
    input_dict = {"tensor": tensor, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    tensor = np.ones((5, 5), dtype=np.float32)
    gain = 1.0
    input_dict = {"tensor": tensor, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    tensor = np.random.rand(3, 5, 2).astype(np.float32)
    gain = 0.5
    input_dict = {"tensor": tensor, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    tensor = np.zeros((4, 6))
    gain = 1.2
    input_dict = {"tensor": tensor, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    tensor = np.random.rand(2, 3).astype(np.float64)
    gain = 2.5
    input_dict = {"tensor": tensor, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.init.xavier_uniform_"] = xavier_uniform_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.init.xavier_uniform_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.init.xavier_uniform_'.")

check_valid('torch.nn.init.xavier_uniform_', generated_inputs['torch.nn.init.xavier_uniform_'], lib="torch", suffix=0)
