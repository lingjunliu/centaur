
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def heaviside_inputs():
    list_of_inputs = []

    # Input 1, valid
    input_tensor = torch.tensor([-1.5, 0, 2.0]).float().numpy()
    values_tensor = torch.tensor([0.5]).float().numpy()
    out_tensor = torch.tensor([0.0, 0.0, 0.0]).float().numpy()

    input_dict = {
        "input": input_tensor,
        "values": values_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid
    input_tensor = torch.tensor([-1.5, 0, 2.0]).float().numpy()
    values_tensor = torch.tensor([1.2, -2.0, 3.5]).float().numpy()
    out_tensor = torch.tensor([0.0, 0.0, 0.0]).float().numpy()

    input_dict = {
        "input": input_tensor,
        "values": values_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid - different dimension
    input_tensor = torch.randn(2, 3).float().numpy()
    values_tensor = torch.randn(1).float().numpy()
    out_tensor = torch.zeros(2, 3).float().numpy()

    input_dict = {
        "input": input_tensor,
        "values": values_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid - all zeros
    input_tensor = torch.zeros(5).float().numpy()
    values_tensor = torch.tensor([2.0]).float().numpy()
    out_tensor = torch.zeros(5).float().numpy()

    input_dict = {
        "input": input_tensor,
        "values": values_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid - all positive
    input_tensor = torch.ones(4).float().numpy()
    values_tensor = torch.tensor([-1.0]).float().numpy()
    out_tensor = torch.zeros(4).float().numpy()

    input_dict = {
        "input": input_tensor,
        "values": values_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, valid - mixed positive and negative
    input_tensor = torch.tensor([-2, -1, 0, 1, 2]).float().numpy()
    values_tensor = torch.tensor([0.0]).float().numpy()
    out_tensor = torch.zeros(5).float().numpy()

    input_dict = {
        "input": input_tensor,
        "values": values_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.heaviside"] = heaviside_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.heaviside' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.heaviside'.")

check_valid('torch.heaviside', generated_inputs['torch.heaviside'], lib="torch", suffix=0)
