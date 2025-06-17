
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import torch.nn as nn
import copy
import numpy as np

def ModuleDict_inputs():
    list_of_inputs = []

    # Input 1: Empty dictionary
    modules = {}
    input_dict = {"modules": modules}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.ModuleDict_1"] = ModuleDict_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.ModuleDict_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ModuleDict_1'.")

check_valid('torch.nn.ModuleDict', generated_inputs['torch.nn.ModuleDict_1'], lib="torch")
