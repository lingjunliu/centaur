
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def script_if_tracing_inputs():
    list_of_inputs = []

    # Input 1: Empty lists
    input_dict = {
        "fn": [],
        "alternative_fn": []
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Simple functions
    def fn1():
        return 1
    def fn2():
        return 2
    input_dict = {
        "fn": [fn1],
        "alternative_fn": [fn2]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multiple functions
    def fn3():
        return 3
    def fn4():
        return 4
    input_dict = {
        "fn": [fn3, fn4],
        "alternative_fn": []
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Functions with arguments
    def fn7(x):
        return x + 7
    def fn8(x):
        return x + 8
    input_dict = {
        "fn": [fn7],
        "alternative_fn": [fn8]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Mixed functions
    def fn9():
        return 9
    def fn10(x):
        return x + 10
    input_dict = {
        "fn": [fn9],
        "alternative_fn": [fn10]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.jit.script_if_tracing"] = script_if_tracing_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.jit.script_if_tracing' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.jit.script_if_tracing'.")

check_valid('torch.jit.script_if_tracing', generated_inputs['torch.jit.script_if_tracing'], lib="torch", suffix=0)
