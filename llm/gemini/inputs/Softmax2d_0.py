
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def Softmax2d_inputs():
    list_of_inputs = []

    input1 = torch.randn(2, 3, 12, 13).numpy()
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(1, 5, 8, 8).numpy()
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(3, 4, 5, 5).numpy()
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(1, 2, 7, 7).numpy()
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = torch.randn(4, 3, 6, 6).numpy()
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.Softmax2d"] = Softmax2d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.Softmax2d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Softmax2d'.")

check_valid('torch.nn.Softmax2d', generated_inputs['torch.nn.Softmax2d'], lib="torch")
