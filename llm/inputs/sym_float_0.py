
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def sym_float_inputs():
    list_of_inputs = []

    # Input 1: Scalar float tensor
    a = torch.tensor(3.14).numpy()
    input_dict = {"a": a.item()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Scalar float tensor with a negative value
    a = torch.tensor(-2.718).numpy()
    input_dict = {"a": a.item()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Scalar float tensor with a 0 value
    a = torch.tensor(0.0).numpy()
    input_dict = {"a": a.item()}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Scalar float tensor with a large value
    a = torch.tensor(1000000.0).numpy()
    input_dict = {"a": a.item()}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Scalar float tensor with a small value
    a = torch.tensor(0.000001).numpy()
    input_dict = {"a": a.item()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.sym_float"] = sym_float_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.sym_float' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sym_float'.")

check_valid('torch.sym_float', generated_inputs['torch.sym_float'], lib="torch")
