
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def dropout2d_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = torch.randn(20, 16, 32, 32).numpy()
    p = 0.2
    inplace = False
    input_dict = {"p": p, "inplace": inplace, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = torch.randn(1, 3, 64, 64).numpy()
    p = 0.7
    inplace = True
    input_dict = {"p": p, "inplace": inplace, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = torch.randn(5, 8, 128, 128).numpy()
    p = 0.0
    inplace = False
    input_dict = {"p": p, "inplace": inplace, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = torch.randn(10, 1, 256, 256).numpy()
    p = 1.0
    inplace = True
    input_dict = {"p": p, "inplace": inplace, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = torch.randn(4, 4, 16, 16).numpy()
    p = 0.5
    inplace = False
    input_dict = {"p": p, "inplace": inplace, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6 (3D input - N, C, L)
    input_tensor = torch.randn(2, 5, 10).numpy()
    p = 0.3
    inplace = True
    input_dict = {"p": p, "inplace": inplace, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 (small input)
    input_tensor = torch.randn(1, 1, 2, 2).numpy()
    p = 0.9
    inplace = False
    input_dict = {"p": p, "inplace": inplace, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = torch.randn(3, 7, 20, 20).numpy()
    p = 0.1
    inplace = True
    input_dict = {"p": p, "inplace": inplace, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 (large input)
    input_tensor = torch.randn(2, 2, 512, 512).numpy()
    p = 0.6
    inplace = False
    input_dict = {"p": p, "inplace": inplace, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 (another 3D input - N, C, L)
    input_tensor = torch.randn(4, 10, 5).numpy()
    p = 0.4
    inplace = True
    input_dict = {"p": p, "inplace": inplace, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.Dropout2d"] = dropout2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.Dropout2d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Dropout2d'.")

check_valid('torch.nn.Dropout2d', generated_inputs['torch.nn.Dropout2d'], lib="torch", suffix=0)
