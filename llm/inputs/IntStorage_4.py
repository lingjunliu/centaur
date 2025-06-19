
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def IntStorage_inputs():
    list_of_inputs = []

    input1 = {"source": [1, 2, 3, 4, 5]}
    list_of_inputs.append(input1)

    input2 = {"source": []}
    list_of_inputs.append(input2)

    input3 = {"source": [-1, 0, 1, -2, 2]}
    list_of_inputs.append(input3)
    
    input4 = {"source": [-(2**31) + 1, 2**31 -1]}
    list_of_inputs.append(input4)

    input5 = {"source": [1, 2, 3, 4]}
    list_of_inputs.append(input5)

    return list_of_inputs

generated_inputs["torch.IntStorage_4"] = IntStorage_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.IntStorage_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.IntStorage_4'.")

check_valid('torch.IntStorage', generated_inputs['torch.IntStorage_4'], lib="torch")
