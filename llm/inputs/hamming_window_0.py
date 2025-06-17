
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def hamming_window_inputs():
    list_of_inputs = []

    input_dict = {
        "window_length": 5,
        "dtype": torch.float32,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "window_length": 10,
        "dtype": torch.float64,
        "requires_grad": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "window_length": 7,
        "periodic": False,
        "alpha": 0.2,
        "dtype": torch.float16,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "window_length": 12,
        "periodic": True,
        "alpha": 0.8,
        "dtype": torch.float32,
        "requires_grad": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "window_length": 3,
        "dtype": torch.float64,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "window_length": 4,
        "periodic": False,
        "alpha": 0.7,
        "beta": 0.3,
        "dtype": torch.float32,
        "requires_grad": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.hamming_window"] = hamming_window_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.hamming_window' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.hamming_window'.")

check_valid('torch.hamming_window', generated_inputs['torch.hamming_window'], lib="torch")
