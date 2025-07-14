
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def kaiming_uniform_inputs():
    list_of_inputs = []

    # Input 1
    tensor = np.zeros((3, 5)).astype(np.float32)
    a = 0.0
    mode = 'fan_in'
    nonlinearity = 'relu'
    input_dict = {"tensor": tensor, "a": a, "mode": mode, "nonlinearity": nonlinearity}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    tensor = np.zeros((2, 3, 4)).astype(np.float32)
    a = np.sqrt(5.0)
    mode = 'fan_out'
    nonlinearity = 'leaky_relu'
    input_dict = {"tensor": tensor, "a": a, "mode": mode, "nonlinearity": nonlinearity}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    tensor = np.zeros((10, 20)).astype(np.float32)
    a = 1.0
    mode = 'fan_in'
    nonlinearity = 'tanh'
    input_dict = {"tensor": tensor, "a": a, "mode": mode, "nonlinearity": nonlinearity}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    tensor = np.zeros((2, 2, 2, 2)).astype(np.float32)
    a = 2.0
    mode = 'fan_out'
    nonlinearity = 'sigmoid'
    input_dict = {"tensor": tensor, "a": a, "mode": mode, "nonlinearity": nonlinearity}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    tensor = np.zeros((1, 100, 100)).astype(np.float32)
    a = 0.1
    mode = 'fan_in'
    nonlinearity = 'relu'
    input_dict = {"tensor": tensor, "a": a, "mode": mode, "nonlinearity": nonlinearity}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    tensor = np.zeros((5, 5)).astype(np.float32)
    a = -0.5
    mode = 'fan_out'
    nonlinearity = 'leaky_relu'
    input_dict = {"tensor": tensor, "a": a, "mode": mode, "nonlinearity": nonlinearity}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    tensor = np.zeros((2, 4, 8, 16)).astype(np.float32)
    a = 3.0
    mode = 'fan_in'
    nonlinearity = 'relu'
    input_dict = {"tensor": tensor, "a": a, "mode": mode, "nonlinearity": nonlinearity}
    list_of_inputs.append(copy.deepcopy(input_dict))


    # Input 8
    tensor = np.zeros((1, 1, 1, 1)).astype(np.float32)
    a = 0.0
    mode = 'fan_in'
    nonlinearity = 'sigmoid'
    input_dict = {"tensor": tensor, "a": a, "mode": mode, "nonlinearity": nonlinearity}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    tensor = np.zeros((2, 3, 4, 5)).astype(np.float32)
    a = 1.0
    mode = 'fan_out'
    nonlinearity = 'relu'
    input_dict = {"tensor": tensor, "a": a, "mode": mode, "nonlinearity": nonlinearity}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    tensor = np.zeros((10, 20)).astype(np.float32)
    a = 2.5
    mode = 'fan_in'
    nonlinearity = 'leaky_relu'
    input_dict = {"tensor": tensor, "a": a, "mode": mode, "nonlinearity": nonlinearity}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.init.kaiming_uniform_"] = kaiming_uniform_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.init.kaiming_uniform_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.init.kaiming_uniform_'.")

check_valid('torch.nn.init.kaiming_uniform_', generated_inputs['torch.nn.init.kaiming_uniform_'], lib="torch", suffix=0)
