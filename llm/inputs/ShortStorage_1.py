
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def ShortStorage_inputs():
    list_of_inputs = []

    input_dict = {
        "size": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "size": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "size": 10
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "size": 100
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "size": 1000
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "size": 2048
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.ShortStorage_1"] = ShortStorage_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.ShortStorage_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.ShortStorage_1'.")

check_valid('torch.ShortStorage', generated_inputs['torch.ShortStorage_1'], lib="torch")
