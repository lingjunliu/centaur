
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import copy
import numpy as np

def histc_inputs():
    list_of_inputs = []

    # Input 1: Basic example with positive values
    input_tensor = torch.tensor([1.0, 2.0, 1.0, 3.0, 2.5]).numpy()
    bins = 5
    min_val = 0.0
    max_val = 4.0
    out_tensor = torch.tensor([]).numpy()

    input_dict = {
        "input": input_tensor,
        "bins": bins,
        "min": min_val,
        "max": max_val,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2:  Negative values and specified range
    input_tensor = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0]).numpy()
    bins = 10
    min_val = -2.0
    max_val = 2.0
    out_tensor = torch.tensor([]).numpy()
    input_dict = {
        "input": input_tensor,
        "bins": bins,
        "min": min_val,
        "max": max_val,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3:  Zero min and max, using data range
    input_tensor = torch.tensor([0.5, 1.5, 2.5, 3.5]).numpy()
    bins = 4
    min_val = 0.0
    max_val = 0.0
    out_tensor = torch.tensor([]).numpy()
    input_dict = {
        "input": input_tensor,
        "bins": bins,
        "min": min_val,
        "max": max_val,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4:  Larger number of bins
    input_tensor = torch.randn(100).numpy()
    bins = 50
    min_val = -3.0
    max_val = 3.0
    out_tensor = torch.tensor([]).numpy()
    input_dict = {
        "input": input_tensor,
        "bins": bins,
        "min": min_val,
        "max": max_val,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5:  Smaller number of bins
    input_tensor = torch.tensor([1.0, 1.5, 2.0, 2.5, 3.0]).numpy()
    bins = 2
    min_val = 1.0
    max_val = 3.0
    out_tensor = torch.tensor([]).numpy()
    input_dict = {
        "input": input_tensor,
        "bins": bins,
        "min": min_val,
        "max": max_val,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.histc"] = histc_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.histc' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.histc'.")

check_valid('torch.histc', generated_inputs['torch.histc'], lib="torch", suffix=0)
