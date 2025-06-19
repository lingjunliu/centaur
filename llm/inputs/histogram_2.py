
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def histogram_inputs():
    list_of_inputs = []

    # Input 1: bins as a tensor
    input_tensor = torch.tensor([1.0, 2.0, 1.0, 3.0, 4.0, 2.0]).numpy()
    bins_tensor = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).numpy()
    range_tuple = (0.0, 5.0)
    weight_tensor = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).numpy()
    density_bool = False
    out_tensor = torch.tensor([]).numpy()

    input_dict = {
        "input": input_tensor,
        "bins": bins_tensor,
        "range": range_tuple,
        "weight": weight_tensor,
        "density": density_bool,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: bins as an integer
    input_tensor = torch.tensor([-1.0, -2.0, -1.0, -3.0, -4.0, -2.0]).numpy()
    bins_tensor = torch.tensor([1,2,3,4,5]).numpy()  # Integer number of bins
    range_tuple = (-5.0, 0.0)
    weight_tensor = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).numpy()
    density_bool = True
    out_tensor = torch.tensor([]).numpy()

    input_dict = {
        "input": input_tensor,
        "bins": bins_tensor,
        "range": range_tuple,
        "weight": weight_tensor,
        "density": density_bool,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different range and weights
    input_tensor = torch.tensor([1.0, 2.0, 1.0, 3.0, 4.0, 2.0]).numpy()
    bins_tensor = torch.tensor([0.0, 2.5, 5.0]).numpy()
    range_tuple = (0.0, 4.0)  # Different range
    weight_tensor = torch.tensor([1.0, 2.0, 1.0, 2.0, 1.0, 2.0]).numpy()  # Different weights
    density_bool = False
    out_tensor = torch.tensor([]).numpy()

    input_dict = {
        "input": input_tensor,
        "bins": bins_tensor,
        "range": range_tuple,
        "weight": weight_tensor,
        "density": density_bool,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative range and density=True
    input_tensor = torch.tensor([-1.0, -2.0, -1.0, -3.0, -4.0, -2.0]).numpy()
    bins_tensor = torch.tensor([-5,-4,-3,-2,-1]).numpy()
    range_tuple = (-5.0, 0.0)
    weight_tensor = torch.tensor([0.5, 1.0, 0.5, 1.0, 0.5, 1.0]).numpy()
    density_bool = True
    out_tensor = torch.tensor([]).numpy()

    input_dict = {
        "input": input_tensor,
        "bins": bins_tensor,
        "range": range_tuple,
        "weight": weight_tensor,
        "density": density_bool,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different input values
    input_tensor = torch.tensor([0.5, 1.5, 2.5, 3.5]).numpy()
    bins_tensor = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]).numpy()
    range_tuple = (0.0, 4.0)
    weight_tensor = torch.tensor([2.0, 1.0, 3.0, 0.5]).numpy()
    density_bool = False
    out_tensor = torch.tensor([]).numpy()

    input_dict = {
        "input": input_tensor,
        "bins": bins_tensor,
        "range": range_tuple,
        "weight": weight_tensor,
        "density": density_bool,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.histogram_2"] = histogram_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.histogram_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.histogram_2'.")

check_valid('torch.histogram', generated_inputs['torch.histogram_2'], lib="torch", suffix=2)
