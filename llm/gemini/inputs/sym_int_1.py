
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def sym_int_inputs():
    list_of_inputs = []

    input1 = np.array(5)
    input_dict1 = {"a": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array(-3)
    input_dict2 = {"a": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array(0)
    input_dict3 = {"a": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.array(2**31 - 1) # Maximum 32-bit integer
    input_dict4 = {"a": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array(-(2**31)) # Minimum 32-bit integer
    input_dict5 = {"a": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs

generated_inputs["torch.sym_int_1"] = sym_int_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.sym_int_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sym_int_1'.")

check_valid('torch.sym_int', generated_inputs['torch.sym_int_1'], lib="torch")
