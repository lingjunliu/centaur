
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import copy
import numpy as np

def autocast_1_inputs():
    list_of_inputs = []
    
    # The error `TypeError: autocast.__call__() missing 1 required positional argument: 'func'`
    # is caused by the test harness's method of invoking the API. `torch.autocast` is a
    # context manager or a decorator. The test harness correctly creates an instance of
    # `torch.autocast` using the provided arguments, but then incorrectly attempts to call
    # that instance as a function. The `__call__` method of an `autocast` instance is
    # for decoration and requires a function as an argument, which is not being provided.
    # The following inputs are valid for the constructor of `torch.autocast` as per the
    # specified signature. The error cannot be fixed by modifying these inputs, as it
    # stems from an incompatibility between the test harness and this specific API's usage pattern.

    # Input 1
    input_dict = {
        'device_type': 'cuda',
        'enabled': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2
    input_dict = {
        'device_type': 'cuda',
        'enabled': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        'device_type': 'cpu',
        'enabled': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4
    input_dict = {
        'device_type': 'cpu',
        'enabled': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        'device_type': 'mps',
        'enabled': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        'device_type': 'mps',
        'enabled': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    input_dict = {
        'device_type': 'xpu',
        'enabled': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    input_dict = {
        'device_type': 'xpu',
        'enabled': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        'device_type': 'hpu',
        'enabled': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    input_dict = {
        'device_type': 'hpu',
        'enabled': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.autocast_1"] = autocast_1_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.autocast_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.autocast_1'.")

check_valid('torch.autocast', generated_inputs['torch.autocast_1'], lib="torch", suffix=1)
