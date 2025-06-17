
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def eye_inputs():
    list_of_inputs = []

    input_dict = {
        "n": 3,
        "out": None,
        "dtype": None,
        "layout": torch.strided,
        "device": None,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "n": 2,
        "m": 4,
        "out": None,
        "dtype": None,
        "layout": torch.strided,
        "device": None,
        "requires_grad": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "n": 4,
        "out": None,
        "dtype": torch.float64,
        "layout": torch.strided,
        "device": None,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "n": 2,
        "m": 3,
        "out": None,
        "dtype": torch.int32,
        "layout": torch.strided,
        "device": None,
        "requires_grad": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.eye"] = eye_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.eye' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.eye'.")

check_valid('torch.eye', generated_inputs['torch.eye'], lib="torch")
