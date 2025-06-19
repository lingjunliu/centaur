
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def randperm_inputs():
    list_of_inputs = []
    
    # Input 1: Basic integer input
    input_dict = {
        "n": 5,
        "generator": None,
        "out": None,
        "dtype": torch.int64,
        "layout": torch.strided,
        "device": None,
        "requires_grad": False,
        "pin_memory": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: Different dtype
    input_dict = {
        "n": 3,
        "generator": None,
        "out": None,
        "dtype": torch.int32,
        "layout": torch.strided,
        "device": None,
        "requires_grad": False,
        "pin_memory": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: With generator
    gen = torch.Generator()
    gen.manual_seed(42)
    input_dict = {
        "n": 7,
        "generator": gen,
        "out": None,
        "dtype": torch.int64,
        "layout": torch.strided,
        "device": None,
        "requires_grad": False,
        "pin_memory": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: With out tensor
    out_tensor = torch.empty(10, dtype=torch.int64)
    input_dict = {
        "n": 10,
        "generator": None,
        "out": out_tensor.numpy(),
        "dtype": torch.int64,
        "layout": torch.strided,
        "device": None,
        "requires_grad": False,
        "pin_memory": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Pin Memory
    input_dict = {
        "n": 8,
        "generator": None,
        "out": None,
        "dtype": torch.int64,
        "layout": torch.strided,
        "device": None,
        "requires_grad": False,
        "pin_memory": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.randperm"] = randperm_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.randperm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.randperm'.")

check_valid('torch.randperm', generated_inputs['torch.randperm'], lib="torch")
