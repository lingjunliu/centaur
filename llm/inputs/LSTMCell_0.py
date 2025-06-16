
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def LSTMCell_inputs():
    list_of_inputs = []

    input_size = 10
    hidden_size = 20

    input = torch.randn(3, input_size).numpy()
    h_0 = torch.randn(3, hidden_size).numpy()
    c_0 = torch.randn(3, hidden_size).numpy()

    input_dict = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "bias": True,
        "input": input,
        "h_0": h_0,
        "c_0": c_0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_size = 5
    hidden_size = 10
    input = torch.randn(input_size).numpy()
    h_0 = torch.randn(hidden_size).numpy()
    c_0 = torch.randn(hidden_size).numpy()

    input_dict = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "bias": False,
        "input": input,
        "h_0": h_0,
        "c_0": c_0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_size = 15
    hidden_size = 25
    input = torch.randn(2, input_size).numpy()
    h_0 = torch.randn(2, hidden_size).numpy()
    c_0 = torch.randn(2, hidden_size).numpy()

    input_dict = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "bias": True,
        "input": input,
        "h_0": h_0,
        "c_0": c_0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_size = 7
    hidden_size = 13
    input = torch.randn(input_size).numpy()
    h_0 = torch.randn(hidden_size).numpy()
    c_0 = torch.randn(hidden_size).numpy()

    input_dict = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "bias": False,
        "input": input,
        "h_0": h_0,
        "c_0": c_0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_size = 3
    hidden_size = 5
    input = torch.randn(4, input_size).numpy()
    h_0 = torch.randn(4, hidden_size).numpy()
    c_0 = torch.randn(4, hidden_size).numpy()

    input_dict = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "bias": True,
        "input": input,
        "h_0": h_0,
        "c_0": c_0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_size = 20
    hidden_size = 30
    input = torch.randn(1, input_size).numpy()
    h_0 = torch.randn(1, hidden_size).numpy()
    c_0 = torch.randn(1, hidden_size).numpy()

    input_dict = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "bias": False,
        "input": input,
        "h_0": h_0,
        "c_0": c_0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.LSTMCell' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.LSTMCell'.")

check_valid('torch.nn.LSTMCell', generated_inputs['torch.nn.LSTMCell'], lib="torch")
