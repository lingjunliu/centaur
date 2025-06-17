
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def GRUCell_inputs():
    list_of_inputs = []

    input_size = 10
    hidden_size = 20

    input1 = torch.randn(3, input_size).numpy()
    hidden1 = torch.randn(3, hidden_size).numpy()
    input_dict1 = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "bias": True,
        "dtype": None,
        "input": input1,
        "hidden": hidden1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(5, input_size).numpy()
    hidden2 = torch.randn(5, hidden_size).numpy()
    input_dict2 = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "bias": False,
        "dtype": None,
        "input": input2,
        "hidden": hidden2,
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input_size = 5
    hidden_size = 10
    input3 = torch.randn(1, input_size).numpy()
    hidden3 = torch.randn(1, hidden_size).numpy()
    input_dict3 = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "bias": True,
        "dtype": None,
        "input": input3,
        "hidden": hidden3,
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input_size = 15
    hidden_size = 25
    input4 = torch.randn(7, input_size).numpy()
    hidden4 = torch.randn(7, hidden_size).numpy()
    input_dict4 = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "bias": False,
        "dtype": None,
        "input": input4,
        "hidden": hidden4,
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input_size = 20
    hidden_size = 30
    input5 = torch.randn(9, input_size).numpy()
    hidden5 = torch.randn(9, hidden_size).numpy()
    input_dict5 = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "bias": True,
        "dtype": None,
        "input": input5,
        "hidden": hidden5,
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.GRUCell"] = GRUCell_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.GRUCell' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.GRUCell'.")

check_valid('torch.nn.GRUCell', generated_inputs['torch.nn.GRUCell'], lib="torch")
