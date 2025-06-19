
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def rfftfreq_inputs():
    list_of_inputs = []

    input_dict = {
        "n": 5,
        "d": 1.0,
        "out": None,
        "dtype": None,
        "layout": torch.strided,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "n": 4,
        "d": 2.0,
        "out": None,
        "dtype": None,
        "layout": torch.strided,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "n": 10,
        "d": 0.5,
        "out": None,
        "dtype": torch.float64,
        "layout": torch.strided,
        "requires_grad": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    out_tensor = torch.empty((10 + 1) // 2) 
    input_dict = {
        "n": 10,
        "d": 1.0,
        "out": out_tensor.numpy(),
        "dtype": None,
        "layout": torch.strided,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "n": 7,
        "d": 3.0,
        "out": None,
        "dtype": torch.float32,
        "layout": torch.strided,
        "requires_grad": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.fft.rfftfreq"] = rfftfreq_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.fft.rfftfreq' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.fft.rfftfreq'.")

check_valid('torch.fft.rfftfreq', generated_inputs['torch.fft.rfftfreq'], lib="torch")
