
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import torch.nn as nn
import numpy as np
import copy

def generate_parameterlist_inputs():
    list_of_inputs = []

    # Input 1: Empty list
    input_dict = {"values": []}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: List of one Parameter
    params = [nn.Parameter(torch.randn(2, 3))]
    input_dict = {"values": params}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: List of multiple Parameters
    params = [nn.Parameter(torch.randn(2, 3)), nn.Parameter(torch.randn(3, 4))]
    input_dict = {"values": params}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: List of Parameters with different shapes
    params = [nn.Parameter(torch.randn(5)), nn.Parameter(torch.randn(2, 2))]
    input_dict = {"values": params}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: List of Parameters with different dtypes
    params = [nn.Parameter(torch.randn(5, dtype=torch.float32))]
    input_dict = {"values": params}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: List with a single large parameter
    params = [nn.Parameter(torch.randn(10, 10))]
    input_dict = {"values": params}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.ParameterList"] = generate_parameterlist_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.ParameterList' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ParameterList'.")

check_valid('torch.nn.ParameterList', generated_inputs['torch.nn.ParameterList'], lib="torch", suffix=0)
