
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def eye_inputs():
    list_of_inputs = []

    # Input 1: Square matrix
    tensor = torch.empty(5, 5).numpy()
    input_dict = {"tensor": tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Rectangular matrix (rows > cols)
    tensor = torch.empty(7, 3).numpy()
    input_dict = {"tensor": tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Rectangular matrix (cols > rows)
    tensor = torch.empty(3, 7).numpy()
    input_dict = {"tensor": tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Small matrix
    tensor = torch.empty(1, 1).numpy()
    input_dict = {"tensor": tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Larger matrix
    tensor = torch.empty(10, 10).numpy()
    input_dict = {"tensor": tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Square matrix with different dimensions
    tensor = torch.empty(2, 2).numpy()
    input_dict = {"tensor": tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Rectangular matrix
    tensor = torch.empty(4, 6).numpy()
    input_dict = {"tensor": tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Another rectangular matrix
    tensor = torch.empty(6, 4).numpy()
    input_dict = {"tensor": tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Different sized square matrix
    tensor = torch.empty(8, 8).numpy()
    input_dict = {"tensor": tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Different sized rectangular matrix
    tensor = torch.empty(9, 5).numpy()
    input_dict = {"tensor": tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Zero sized matrix
    tensor = torch.empty(0, 0).numpy()
    input_dict = {"tensor": tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Non-square 0 dimension matrix
    tensor = torch.empty(0, 5).numpy()
    input_dict = {"tensor": tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.init.eye_"] = eye_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.init.eye_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.init.eye_'.")

check_valid('torch.nn.init.eye_', generated_inputs['torch.nn.init.eye_'], lib="torch", suffix=0)
