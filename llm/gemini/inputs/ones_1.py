
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np
from torch import float32, int64, float64, int32, float16

def ones_inputs():
    list_of_inputs = []

    # Input 1: Basic case with size and default values
    size = 5
    out = torch.empty(5).numpy()
    dtype = float32
    requires_grad = False

    input_dict = {
        "size": size,
        "out": out,
        "dtype": dtype,
        "layout": torch.strided,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Multi-dimensional tensor
    size = (2, 3)
    out = torch.empty(2,3).numpy()
    dtype = int64
    requires_grad = True

    input_dict = {
        "size": size,
        "out": out,
        "dtype": dtype,
        "layout": torch.strided,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Using a list for size
    size = [4, 2, 1]
    out = torch.empty(4,2,1).numpy()
    dtype = float64
    requires_grad = False

    input_dict = {
        "size": size,
        "out": out,
        "dtype": dtype,
        "layout": torch.strided,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Different out tensor
    size = 1
    out = torch.empty(1, dtype=torch.int32).numpy()
    dtype = int32
    requires_grad = True

    input_dict = {
        "size": size,
        "out": out,
        "dtype": dtype,
        "layout": torch.strided,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Specify device
    size = 3
    out = torch.empty(3).numpy()
    dtype = float16
    requires_grad = False

    input_dict = {
        "size": size,
        "out": out,
        "dtype": dtype,
        "layout": torch.strided,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    

    return list_of_inputs

generated_inputs["torch.ones_1"] = ones_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.ones_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.ones_1'.")

check_valid('torch.ones', generated_inputs['torch.ones_1'], lib="torch", suffix=1)
