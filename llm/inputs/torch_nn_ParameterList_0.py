
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np
import torch.nn as nn

def ParameterList_inputs():
    list_of_inputs = []

    def create_numpy_parameter(shape):
        return nn.Parameter(torch.randn(shape)).detach().numpy()

    # Input 1: Empty list
    input_dict = {"values": []}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: List of single Parameter
    input_dict = {"values": [create_numpy_parameter((2, 3))]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: List of multiple Parameters
    input_dict = {"values": [create_numpy_parameter((2, 3)), create_numpy_parameter((3, 4))]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: List with Parameters of different shapes
    input_dict = {"values": [create_numpy_parameter((5,)), create_numpy_parameter((2, 2))]}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: List with different dtypes
    input_dict = {"values": [nn.Parameter(torch.randn(2, 3, dtype=torch.float32)).detach().numpy(), nn.Parameter(torch.randn(3, 4, dtype=torch.float64)).detach().numpy()]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: List with different ranges of values
    input_dict = {"values": [nn.Parameter(torch.rand(5)).detach().numpy(), nn.Parameter(torch.randn(2, 2)).detach().numpy()]}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: List with zero and one tensors
    input_dict = {"values": [nn.Parameter(torch.zeros(2, 2)).detach().numpy(), nn.Parameter(torch.ones(3,3)).detach().numpy()]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: List of parameters with a large number of elements
    input_dict = {"values": [create_numpy_parameter((10, 10)) for _ in range(3)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: List of parameters with 1, 2, 3 dimensions
    input_dict = {"values": [create_numpy_parameter((5,)), create_numpy_parameter((2, 2)), create_numpy_parameter((1, 2, 3))]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: List of parameters with a negative range
    input_dict = {"values": [nn.Parameter(torch.randn(2, 2) * 1000).detach().numpy()]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.ParameterList"] = ParameterList_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.ParameterList' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ParameterList'.")

check_valid('torch.nn.ParameterList', generated_inputs['torch.nn.ParameterList'], lib="torch", suffix=0)
