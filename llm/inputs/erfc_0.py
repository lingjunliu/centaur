
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def erfc_inputs():
    list_of_inputs = []

    # Input 1, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    out = torch.tensor([0.0, 0.0, 0.0]).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid, negative input
    input = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    out = torch.tensor([0.0, 0.0, 0.0]).numpy()
    input_dict = {
        "input": input,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid, 2D input
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    out = torch.tensor([[0.0, 0.0], [0.0, 0.0]]).numpy()
    input_dict = {
        "input": input,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid, 3D input
    input = torch.randn(2, 3, 4).numpy()
    out = torch.zeros(2, 3, 4).numpy()
    input_dict = {
        "input": input,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid, zero input
    input = torch.tensor([0.0, 0.0, 0.0]).numpy()
    out = torch.tensor([0.0, 0.0, 0.0]).numpy()
    input_dict = {
        "input": input,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.erfc"] = erfc_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.erfc' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.erfc'.")

check_valid('torch.erfc', generated_inputs['torch.erfc'], lib="torch", suffix=0)
