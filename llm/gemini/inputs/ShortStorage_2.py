
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def ShortStorage_inputs():
    list_of_inputs = []
    
    size = [0]
    input_dict = {
        "size": size
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    size = [1]
    input_dict = {
        "size": size
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    size = [5]
    input_dict = {
        "size": size
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    size = [10]
    input_dict = {
        "size": size
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    size = [100]
    input_dict = {
        "size": size
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    size = [1000]
    input_dict = {
        "size": size
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.ShortStorage_2"] = ShortStorage_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.ShortStorage_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.ShortStorage_2'.")

check_valid('torch.ShortStorage', generated_inputs['torch.ShortStorage_2'], lib="torch")
