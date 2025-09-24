
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def vdot_inputs():
    list_of_inputs = []

    # Input 1: Basic valid input with positive integers
    input1 = torch.tensor([1, 2, 3], dtype=torch.float32)
    input2 = torch.tensor([4, 5, 6], dtype=torch.float32)
    out = torch.tensor([], dtype=torch.float32)

    input_dict = {
        "input": input1.numpy(),
        "other": input2.numpy(),
        "out": out.numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Valid input with negative integers
    input1 = torch.tensor([-1, -2, -3], dtype=torch.float32)
    input2 = torch.tensor([4, 5, 6], dtype=torch.float32)
    out = torch.tensor([], dtype=torch.float32)
    input_dict = {
        "input": input1.numpy(),
        "other": input2.numpy(),
        "out": out.numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Valid input with floating-point numbers
    input1 = torch.tensor([1.5, 2.5, 3.5], dtype=torch.float32)
    input2 = torch.tensor([4.5, 5.5, 6.5], dtype=torch.float32)
    out = torch.tensor([], dtype=torch.float32)
    input_dict = {
        "input": input1.numpy(),
        "other": input2.numpy(),
        "out": out.numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Valid input with zeros
    input1 = torch.tensor([0, 0, 0], dtype=torch.float32)
    input2 = torch.tensor([1, 2, 3], dtype=torch.float32)
    out = torch.tensor([], dtype=torch.float32)
    input_dict = {
        "input": input1.numpy(),
        "other": input2.numpy(),
        "out": out.numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Valid input with mixed positive and negative values
    input1 = torch.tensor([-1, 2, -3], dtype=torch.float32)
    input2 = torch.tensor([4, -5, 6], dtype=torch.float32)
    out = torch.tensor([], dtype=torch.float32)
    input_dict = {
        "input": input1.numpy(),
        "other": input2.numpy(),
        "out": out.numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Valid input with complex numbers
    input1 = torch.tensor([1 + 2j, 3 - 1j], dtype=torch.complex64)
    input2 = torch.tensor([2 + 1j, 4 - 0j], dtype=torch.complex64)
    out = torch.tensor([], dtype=torch.complex64)
    input_dict = {
        "input": input1.numpy(),
        "other": input2.numpy(),
        "out": out.numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    

    return list_of_inputs

generated_inputs["torch.vdot"] = vdot_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.vdot' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.vdot'.")

check_valid('torch.vdot', generated_inputs['torch.vdot'], lib="torch", suffix=0)
