
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def maxpool2d_inputs():
    list_of_inputs = []

    input_dict1 = {
        'kernel_size': (2, 2),
        'stride': (2, 2),
        'padding': (0, 0),
        'dilation': (1, 1),
        'return_indices': True,
        'ceil_mode': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input_dict2 = {
        'kernel_size': (2, 2),
        'stride': (1, 1),
        'padding': (0, 0),
        'dilation': (1, 1),
        'return_indices': False,
        'ceil_mode': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    return list_of_inputs

generated_inputs["torch.nn.MaxPool2d"] = maxpool2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.MaxPool2d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.MaxPool2d'.")


check_valid('torch.nn.MaxPool2d', generated_inputs['torch.nn.MaxPool2d'], lib="torch", suffix=0)
