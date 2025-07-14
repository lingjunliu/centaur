
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def remove_weight_norm_inputs():
    list_of_inputs = []

    # Input 1
    module = torch.nn.Linear(5, 10)
    torch.nn.utils.weight_norm(module, name='weight')
    input_dict = {'module': module, 'name': 'weight'}
    list_of_inputs.append(input_dict)

    # Input 2
    module = torch.nn.Conv2d(3, 16, kernel_size=3)
    torch.nn.utils.weight_norm(module, name='weight')
    input_dict = {'module': module, 'name': 'weight'}
    list_of_inputs.append(input_dict)

    # Input 3
    module = torch.nn.LSTM(10, 20)
    torch.nn.utils.weight_norm(module, name='weight_ih_l0')
    input_dict = {'module': module, 'name': 'weight_ih_l0'}
    list_of_inputs.append(input_dict)

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.utils.remove_weight_norm"] = remove_weight_norm_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.utils.remove_weight_norm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.utils.remove_weight_norm'.")

check_valid('torch.nn.utils.remove_weight_norm', generated_inputs['torch.nn.utils.remove_weight_norm'], lib="torch", suffix=0)
