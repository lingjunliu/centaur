
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def set_num_threads_inputs():
    list_of_inputs = []

    input_dict = {
        "threads": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "threads": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "threads": 4
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "threads": 8
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "threads": 16
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.set_num_threads"] = set_num_threads_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.set_num_threads' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.set_num_threads'.")

check_valid('torch.set_num_threads', generated_inputs['torch.set_num_threads'], lib="torch")
