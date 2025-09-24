
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def linspace_inputs():
    list_of_inputs = []

    input_dict = {
        "start": 3.0,
        "end": 10.0,
        "steps": 5,
        "out": None,
        "dtype": None,
        "layout": torch.strided,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "start": -10.0,
        "end": 10.0,
        "steps": 5,
        "out": None,
        "dtype": None,
        "layout": torch.strided,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "start": -10.0,
        "end": 10.0,
        "steps": 1,
        "out": None,
        "dtype": None,
        "layout": torch.strided,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "start": -5.0,
        "end": 5.0,
        "steps": 10,
        "out": None,
        "dtype": torch.float32,
        "layout": torch.strided,
        "requires_grad": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "start": 0.0,
        "end": 1.0,
        "steps": 100,
        "out": None,
        "dtype": torch.float64,
        "layout": torch.strided,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.linspace"] = linspace_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.linspace' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linspace'.")

check_valid('torch.linspace', generated_inputs['torch.linspace'], lib="torch")
