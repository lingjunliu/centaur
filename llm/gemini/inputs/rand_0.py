
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy
import numpy as np

def rand_inputs():
    list_of_inputs = []

    # Input 1: Basic size
    input_dict = {"size": 4, "generator": None, "out": None, "dtype": None, "layout": torch.strided, "device": None, "requires_grad": False, "pin_memory": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Tuple size
    input_dict = {"size": (2, 3), "generator": None, "out": None, "dtype": None, "layout": torch.strided, "device": None, "requires_grad": False, "pin_memory": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: List size, specific dtype
    input_dict = {"size": [5, 2], "generator": None, "out": None, "dtype": torch.float64, "layout": torch.strided, "device": None, "requires_grad": True, "pin_memory": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Generator
    gen = torch.Generator()
    input_dict = {"size": (1, 4, 3), "generator": gen, "out": None, "dtype": None, "layout": torch.strided, "device": None, "requires_grad": False, "pin_memory": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Out tensor
    out_tensor = torch.empty(2, 2)
    input_dict = {"size": (2, 2), "generator": None, "out": out_tensor, "dtype": None, "layout": torch.strided, "device": None, "requires_grad": False, "pin_memory": False}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Different device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    input_dict = {"size": (3, 1), "generator": None, "out": None, "dtype": None, "layout": torch.strided, "device": device, "requires_grad": False, "pin_memory": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.rand"] = rand_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.rand' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.rand'.")

check_valid('torch.rand', generated_inputs['torch.rand'], lib="torch")
