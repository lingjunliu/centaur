
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def QFunctional_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        "inner": {
            "args": [torch.randn(3, 4).numpy()],
            "kwargs": {}
        }
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "inner": {
            "args": [torch.randn(2, 2, 2).numpy()],
            "kwargs": {}
        }
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "inner": {
            "args": [torch.randn(1, 5).numpy()],
            "kwargs": {}
        }
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4
    input_dict = {
        "inner": {
            "args": [torch.randn(4).numpy()],
            "kwargs": {}
        }
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    input_dict = {
        "inner": {
            "args": [torch.randn(1, 1, 1, 1).numpy()],
            "kwargs": {}
        }
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        "inner": {
            "args": [torch.zeros(2, 3).numpy()],
            "kwargs": {}
        }
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        "inner": {
            "args": [torch.ones(5).numpy()],
            "kwargs": {}
        }
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        "inner": {
            "args": [torch.randint(-5, 5, (3, 3)).float().numpy()],
            "kwargs": {}
        }
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        "inner": {
            "args": [torch.full((2, 4), 3.14).numpy()],
            "kwargs": {}
        }
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        "inner": {
            "args": [torch.arange(10).reshape(2, 5).float().numpy()],
            "kwargs": {}
        }
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.quantized.QFunctional"] = QFunctional_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.quantized.QFunctional' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.quantized.QFunctional'.")

check_valid('torch.nn.quantized.QFunctional', generated_inputs['torch.nn.quantized.QFunctional'], lib="torch", suffix=0)
