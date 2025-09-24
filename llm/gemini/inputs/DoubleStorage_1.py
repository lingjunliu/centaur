
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def DoubleStorage_inputs():
    list_of_inputs = []

    # Input 1: Size of the storage
    size1 = 5
    input_dict1 = {}
    input_dict1["size"] = size1
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Size of the storage
    size2 = 1
    input_dict2 = {}
    input_dict2["size"] = size2
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Size of the storage
    size3 = 10
    input_dict3 = {}
    input_dict3["size"] = size3
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Size of the storage
    size4 = 0
    input_dict4 = {}
    input_dict4["size"] = size4
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5: Size of the storage
    size5 = 2
    input_dict5 = {}
    input_dict5["size"] = size5
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.DoubleStorage_1"] = DoubleStorage_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.DoubleStorage_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.DoubleStorage_1'.")

check_valid('torch.DoubleStorage', generated_inputs['torch.DoubleStorage_1'], lib="torch")
