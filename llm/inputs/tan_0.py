
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def tan_inputs():
    list_of_inputs = []

    # Input 1: 1D tensor with positive and negative floats
    input_1 = torch.randn(5).numpy()
    input_dict_1 = {"input": input_1}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 2D tensor with positive and negative floats
    input_2 = torch.randn(3, 4).numpy()
    input_dict_2 = {"input": input_2}
    list_of_inputs.append(copy.deepcopy(input_dict_2))


    return list_of_inputs

generated_inputs["torch.tan"] = tan_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.tan' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.tan'.")

check_valid('torch.tan', generated_inputs['torch.tan'], lib="torch")
