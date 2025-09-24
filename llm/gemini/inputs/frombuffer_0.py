
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import array
import numpy as np

def frombuffer_inputs():
    list_of_inputs = []

    # Example 1: Basic example with int32
    a = array.array('i', [1, 2, 3, 4, 5])
    input_dict = {
        "buffer": a,
        "dtype": torch.int32,
        "count": -1,
        "offset": 0,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2: Using offset and a smaller count
    b = array.array('f', [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    input_dict = {
        "buffer": b,
        "dtype": torch.float32,
        "count": 3,
        "offset": 4,
        "requires_grad": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 3: Using a different dtype (char) and negative values
    c = array.array('b', [-1, 0, 1, 2, -3, 4])
    input_dict = {
        "buffer": c,
        "dtype": torch.int8,
        "count": -1,
        "offset": 0,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 4: Interpreting bytes as int32 with offset
    d = array.array('b', [-1, 0, 0, 0, 1, 0, 0, 0])
    input_dict = {
        "buffer": d,
        "dtype": torch.int32,
        "count": 1,
        "offset": 0,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 5: Larger array with double and specific count
    e = array.array('d', [i * 0.5 for i in range(10)])
    input_dict = {
        "buffer": e,
        "dtype": torch.float64,
        "count": 5,
        "offset": 8,
        "requires_grad": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.frombuffer"] = frombuffer_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.frombuffer' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.frombuffer'.")

check_valid('torch.frombuffer', generated_inputs['torch.frombuffer'], lib="torch")
