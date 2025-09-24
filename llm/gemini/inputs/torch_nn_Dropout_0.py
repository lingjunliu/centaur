
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def dropout_inputs():
    list_of_inputs = []

    # Input 1
    p = 0.2
    inplace = False
    input_tensor = torch.randn(20, 16).numpy()
    input_dict = {"p": p, "inplace": inplace, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    p = 0.5
    inplace = True
    input_tensor = torch.randn(10, 5, 2).numpy()
    input_dict = {"p": p, "inplace": inplace, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    p = 0.0
    inplace = False
    input_tensor = torch.randn(1, 1, 1, 1).numpy()
    input_dict = {"p": p, "inplace": inplace, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    p = 1.0
    inplace = True
    input_tensor = torch.randn(5).numpy()
    input_dict = {"p": p, "inplace": inplace, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    p = 0.75
    inplace = False
    input_tensor = torch.randn(2, 2, 2, 2, 2).numpy()
    input_dict = {"p": p, "inplace": inplace, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    p = 0.3
    inplace = True
    input_tensor = torch.randn(1).numpy()
    input_dict = {"p": p, "inplace": inplace, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    p = 0.9
    inplace = False
    input_tensor = torch.randn(3, 7).numpy()
    input_dict = {"p": p, "inplace": inplace, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    p = 0.1
    inplace = True
    input_tensor = torch.randn(4, 4, 4).numpy()
    input_dict = {"p": p, "inplace": inplace, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    p = 0.6
    inplace = False
    input_tensor = torch.randn(100).numpy()
    input_dict = {"p": p, "inplace": inplace, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    p = 0.4
    inplace = True
    input_tensor = torch.randn(2, 8, 16).numpy()
    input_dict = {"p": p, "inplace": inplace, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    p = 0.8
    inplace = False
    input_tensor = torch.randn(5, 5).numpy()
    input_dict = {"p": p, "inplace": inplace, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.Dropout"] = dropout_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.Dropout' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Dropout'.")

check_valid('torch.nn.Dropout', generated_inputs['torch.nn.Dropout'], lib="torch", suffix=0)
