
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def swapdims_inputs():
    list_of_inputs = []

    x = torch.randn(2, 3, 4).numpy()
    input_dict = {
        "input": x,
        "dim0": 0,
        "dim1": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = torch.randint(0, 10, (5, 2, 3)).numpy()
    input_dict = {
        "input": x,
        "dim0": 1,
        "dim1": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = torch.randn(2, 2, 2, 2).numpy()
    input_dict = {
        "input": x,
        "dim0": 0,
        "dim1": 3
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = torch.randn(4, 5).numpy()
    input_dict = {
        "input": x,
        "dim0": 0,
        "dim1": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = torch.randn(3, 4, 5, 6).numpy()
    input_dict = {
        "input": x,
        "dim0": 1,
        "dim1": 3
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = torch.randn(3, 4, 5).numpy()
    input_dict = {
        "input": x,
        "dim0": 0,
        "dim1": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = torch.randn(2, 3, 4).numpy()
    input_dict = {
        "input": x,
        "dim0": -3,
        "dim1": -2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = torch.randn(5).numpy()
    input_dict = {
        "input": x,
        "dim0": 0,
        "dim1": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.swapdims"] = swapdims_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.swapdims' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.swapdims'.")

check_valid('torch.swapdims', generated_inputs['torch.swapdims'], lib="torch")
