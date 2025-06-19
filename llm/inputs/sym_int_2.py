
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def sym_int_inputs():
    list_of_inputs = []

    a = np.array(5)
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array(-3)
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([5])
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([-3])
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array(0)
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.sym_int_2"] = sym_int_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.sym_int_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sym_int_2'.")

check_valid('torch.sym_int', generated_inputs['torch.sym_int_2'], lib="torch")
