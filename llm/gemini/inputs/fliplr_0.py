
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def fliplr_inputs():
    list_of_inputs = []

    x = torch.arange(4).view(2, 2).numpy()
    input_dict = {"input": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = torch.randn(3, 3).numpy()
    input_dict = {"input": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = torch.randint(0, 10, (4, 5)).numpy()
    input_dict = {"input": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = torch.randn(2, 3, 4).numpy()
    input_dict = {"input": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = torch.randint(-5, 5, (2, 2)).numpy()
    input_dict = {"input": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = torch.randn(5, 5).double().numpy()
    input_dict = {"input": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.fliplr"] = fliplr_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.fliplr' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.fliplr'.")

check_valid('torch.fliplr', generated_inputs['torch.fliplr'], lib="torch")
