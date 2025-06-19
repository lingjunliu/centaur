
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def expit_inputs():
    list_of_inputs = []

    # Input 1, valid: 1D tensor, positive values
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    out = np.zeros_like(input)

    input_dict = {
        "input": input,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid: 1D tensor, negative values
    input = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    out = np.zeros_like(input)
    input_dict = {
        "input": input,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid: 2D tensor, mixed positive and negative values
    input = torch.tensor([[-1.0, 2.0], [-3.0, 4.0]]).numpy()
    out = np.zeros_like(input)
    input_dict = {
        "input": input,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid: 3D tensor, all zeros
    input = torch.zeros((2, 2, 2)).numpy()
    out = np.zeros_like(input)
    input_dict = {
        "input": input,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid: 1D tensor, large values
    input = torch.tensor([100.0, -100.0, 0.0]).numpy()
    out = np.zeros_like(input)
    input_dict = {
        "input": input,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.special.expit"] = expit_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.special.expit' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.expit'.")

check_valid('torch.special.expit', generated_inputs['torch.special.expit'], lib="torch", suffix=0)
