
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def acosh_inputs():
    list_of_inputs = []

    # Input 1: Basic valid input
    input_tensor = torch.tensor([1.5, 2.0, 2.5]).numpy()
    out_tensor = torch.tensor([0.0, 0.0, 0.0]).numpy()

    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different values
    input_tensor = torch.tensor([1.0, 3.0, 5.0]).numpy()
    out_tensor = torch.tensor([0.0, 0.0, 0.0]).numpy()
    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multi-dimensional tensor
    input_tensor = torch.tensor([[1.1, 1.2], [1.3, 1.4]]).numpy()
    out_tensor = torch.tensor([[0.0, 0.0], [0.0, 0.0]]).numpy()
    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Larger tensor
    input_tensor = torch.rand(2, 3, 4).uniform_(1, 2).numpy()
    out_tensor = torch.zeros(2, 3, 4).numpy()
    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Input with a single element
    input_tensor = torch.tensor([2.0]).numpy()
    out_tensor = torch.tensor([0.0]).numpy()
    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.acosh"] = acosh_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.acosh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.acosh'.")

check_valid('torch.acosh', generated_inputs['torch.acosh'], lib="torch", suffix=0)
