
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def xavier_normal_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D tensor, gain = 1.0
    tensor = torch.empty(3, 5).numpy()
    gain = 1.0
    input_dict = {"tensor": tensor, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D tensor, gain = 0.5
    tensor = torch.empty(10, 2).numpy()
    gain = 0.5
    input_dict = {"tensor": tensor, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D tensor, gain = 2.0
    tensor = torch.empty(2, 3, 4).numpy()
    gain = 2.0
    input_dict = {"tensor": tensor, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Large tensor, gain = 0.1
    tensor = torch.empty(100, 100).numpy()
    gain = 0.1
    input_dict = {"tensor": tensor, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D tensor, gain = 1.5
    tensor = torch.empty(2, 2, 2, 2).numpy()
    gain = 1.5
    input_dict = {"tensor": tensor, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D tensor with small dimensions, gain = 0.8
    tensor = torch.empty(2, 2).numpy()
    gain = 0.8
    input_dict = {"tensor": tensor, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 5D tensor, gain = 1.2
    tensor = torch.empty(1, 2, 3, 4, 5).numpy()
    gain = 1.2
    input_dict = {"tensor": tensor, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D tensor with irregular dimensions, gain = 1.7
    tensor = torch.empty(7, 11).numpy()
    gain = 1.7
    input_dict = {"tensor": tensor, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D tensor with varying dimensions, gain = 0.3
    tensor = torch.empty(4, 8, 12).numpy()
    gain = 0.3
    input_dict = {"tensor": tensor, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Gain close to zero
    tensor = torch.empty(5, 5).numpy()
    gain = 0.001
    input_dict = {"tensor": tensor, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.init.xavier_normal_"] = xavier_normal_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.init.xavier_normal_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.init.xavier_normal_'.")

check_valid('torch.nn.init.xavier_normal_', generated_inputs['torch.nn.init.xavier_normal_'], lib="torch", suffix=0)
