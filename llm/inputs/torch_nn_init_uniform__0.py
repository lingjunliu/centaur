
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def uniform_inputs():
    list_of_inputs = []

    # Input 1
    tensor = np.empty((3, 4))
    a = 0.0
    b = 1.0
    input_dict = {"tensor": tensor, "a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    tensor = np.empty((2, 2, 2))
    a = -1.0
    b = 1.0
    input_dict = {"tensor": tensor, "a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    tensor = np.empty((10,))
    a = 5.0
    b = 10.0
    input_dict = {"tensor": tensor, "a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    tensor = np.empty((1, 1, 1, 1))
    a = -5.0
    b = -2.0
    input_dict = {"tensor": tensor, "a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    tensor = np.empty((5, 5))
    a = 0.0
    b = 0.0
    input_dict = {"tensor": tensor, "a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    tensor = np.empty((2, 3, 4, 5))
    a = -0.5
    b = 0.5
    input_dict = {"tensor": tensor, "a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    tensor = np.empty((1,))
    a = 100.0
    b = 200.0
    input_dict = {"tensor": tensor, "a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    tensor = np.empty((3, 1))
    a = -2.5
    b = 0.0
    input_dict = {"tensor": tensor, "a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    tensor = np.empty((1, 5, 1))
    a = 0.0
    b = 1000.0
    input_dict = {"tensor": tensor, "a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    tensor = np.empty((4, 4, 4, 4))
    a = -10.0
    b = -5.0
    input_dict = {"tensor": tensor, "a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    tensor = np.empty((2,))
    a = 1.0
    b = 1.0
    input_dict = {"tensor": tensor, "a": a, "b": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.init.uniform_"] = uniform_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.init.uniform_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.init.uniform_'.")

check_valid('torch.nn.init.uniform_', generated_inputs['torch.nn.init.uniform_'], lib="torch", suffix=0)
