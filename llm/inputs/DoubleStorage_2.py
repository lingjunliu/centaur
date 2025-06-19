
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def DoubleStorage_inputs():
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
        "size": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.DoubleStorage_2"] = DoubleStorage_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.DoubleStorage_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.DoubleStorage_2'.")

check_valid('torch.DoubleStorage', generated_inputs['torch.DoubleStorage_2'], lib="torch")
