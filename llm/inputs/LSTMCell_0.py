
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def LSTMCell_inputs():
    list_of_inputs = []

    input_size = 10
    hidden_size = 20

    # Input 1: Basic case with bias
    input_val = torch.randn(5, input_size).numpy()
    h_0_val = torch.randn(5, hidden_size).numpy()
    c_0_val = torch.randn(5, hidden_size).numpy()

    input_dict = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "bias": True,
        "input": input_val,
        "h_0": (h_0_val, c_0_val)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: No bias
    input_val = torch.randn(3, input_size).numpy()
    h_0_val = torch.randn(3, hidden_size).numpy()
    c_0_val = torch.randn(3, hidden_size).numpy()

    input_dict = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "bias": False,
        "input": input_val,
        "h_0": (h_0_val, c_0_val)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Single input, h_0 and c_0 provided
    input_val = torch.randn(input_size).numpy()
    h_0_val = torch.randn(hidden_size).numpy()
    c_0_val = torch.randn(hidden_size).numpy()
    input_dict = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "bias": True,
        "input": input_val,
        "h_0": (h_0_val, c_0_val)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: No initial states provided
    input_val = torch.randn(2, input_size).numpy()

    input_dict = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "bias": True,
        "input": input_val,
        "h_0": None,
        "c_0": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different input and hidden size values, single input
    input_size = 5
    hidden_size = 15
    input_val = torch.randn(input_size).numpy()
    h_0_val = torch.randn(hidden_size).numpy()
    c_0_val = torch.randn(hidden_size).numpy()
    input_dict = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "bias": True,
        "input": input_val,
        "h_0": (h_0_val, c_0_val)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.LSTMCell"] = LSTMCell_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.LSTMCell' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.LSTMCell'.")

check_valid('torch.nn.LSTMCell', generated_inputs['torch.nn.LSTMCell'], lib="torch")
