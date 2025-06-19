
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def sym_fresh_size_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        "kwargs": {"expr": "i0"},
        "args": []
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "kwargs": {"expr": "i1 + i2"},
        "args": []
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "kwargs": {"expr": "i3 * 4"},
        "args": []
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "kwargs": {"expr": "i4 // 5"},
        "args": []
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        "kwargs": {"expr": "(i5 + i6) * i7"},
        "args": []
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.sym_fresh_size"] = sym_fresh_size_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.sym_fresh_size' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sym_fresh_size'.")

check_valid('torch.sym_fresh_size', generated_inputs['torch.sym_fresh_size'], lib="torch", suffix=0)
