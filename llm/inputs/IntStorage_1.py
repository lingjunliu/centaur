
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy
import numpy as np

def IntStorage_inputs():
    list_of_inputs = []

    input1 = {
        "size": 5
    }
    list_of_inputs.append(copy.deepcopy(input1))

    input2 = {
        "size": 0
    }
    list_of_inputs.append(copy.deepcopy(input2))

    input3 = {
        "size": 1
    }
    list_of_inputs.append(copy.deepcopy(input3))

    input4 = {
        "size": 100
    }
    list_of_inputs.append(copy.deepcopy(input4))
    
    input5 = {
        "size": 2**10
    }
    list_of_inputs.append(copy.deepcopy(input5))

    input6 = {
        "size": 2**20
    }
    list_of_inputs.append(copy.deepcopy(input6))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.IntStorage_1"] = IntStorage_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.IntStorage_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.IntStorage_1'.")

check_valid('torch.IntStorage', generated_inputs['torch.IntStorage_1'], lib="torch")
