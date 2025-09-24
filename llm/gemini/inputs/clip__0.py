
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def clip_inputs():
    list_of_inputs = []

    # Input 1, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    min = 1.5
    max = 2.5
    input_dict = {
        "input": input,
        "min": min,
        "max": max
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid, negative values
    input = torch.tensor([-1.0, 0.0, 1.0]).numpy()
    min = -0.5
    max = 0.5
    input_dict = {
        "input": input,
        "min": min,
        "max": max
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid, multi-dimensional
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    min = 1.5
    max = 3.5
    input_dict = {
        "input": input,
        "min": min,
        "max": max
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid, different range
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    min = 0.0
    max = 10.0
    input_dict = {
        "input": input,
        "min": min,
        "max": max
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid, min > max (should still work but give all max values)
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    min = 5.0
    max = 0.0
    input_dict = {
        "input": input,
        "min": min,
        "max": max
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.clip_"] = clip_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.clip_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.clip_'.")

check_valid('torch.clip_', generated_inputs['torch.clip_'], lib="torch", suffix=0)
