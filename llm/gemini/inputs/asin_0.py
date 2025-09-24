
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def asin_inputs():
    list_of_inputs = []

    # Input 1: Basic valid input with out
    input_tensor = torch.tensor([-0.5, 0, 0.5]).numpy()
    out_tensor = torch.tensor([0.0, 0.0, 0.0]).numpy()

    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different values, different out tensor size
    input_tensor = torch.tensor([-1.0, -0.7, 0.2, 0.9, 1.0]).numpy()
    out_tensor = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0]).numpy()

    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multidimensional array
    input_tensor = torch.tensor([[-0.8, 0.3], [0.6, -0.1]]).numpy()
    out_tensor = torch.tensor([[0.0, 0.0], [0.0, 0.0]]).numpy()

    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Edge case - values outside the domain [-1, 1] will result in NaN
    input_tensor = torch.tensor([-1.2, 0, 1.5]).numpy()
    out_tensor = torch.tensor([0.0, 0.0, 0.0]).numpy()

    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Larger tensor
    input_tensor = torch.randn(2, 3, 4).numpy()
    out_tensor = torch.zeros(2, 3, 4).numpy()
    input_tensor = np.clip(input_tensor, -1, 1)

    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Input with all zeros
    input_tensor = torch.zeros(5).numpy()
    out_tensor = torch.zeros(5).numpy()

    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Input with all ones
    input_tensor = torch.ones(5).numpy()
    out_tensor = torch.zeros(5).numpy()

    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.asin"] = asin_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.asin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.asin'.")

check_valid('torch.asin', generated_inputs['torch.asin'], lib="torch", suffix=0)
