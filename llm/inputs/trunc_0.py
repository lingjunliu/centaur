
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy
import numpy as np

def trunc_inputs():
    list_of_inputs = []

    input1 = torch.randn(4).numpy()
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(2, 3).numpy()
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randint(-5, 5, (3, 2, 4)).float().numpy()
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.tensor([-3.14, -2.7, -1.0, 0.0, 1.0, 2.7, 3.14]).double().numpy()
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    return list_of_inputs

generated_inputs["torch.trunc"] = trunc_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.trunc' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.trunc'.")

check_valid('torch.trunc', generated_inputs['torch.trunc'], lib="torch")
