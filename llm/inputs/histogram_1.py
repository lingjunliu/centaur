
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_histogram_inputs():
    list_of_inputs = []

    # Input 1: Basic example with integer bins
    input_tensor = torch.tensor([1., 2, 1]).numpy()
    bins = 4
    range_val = (0., 3.)
    weight = torch.tensor([1., 2., 4.]).numpy()
    density = False
    out = None
    input_dict = {
        "input": input_tensor,
        "bins": bins,
        "range": range_val,
        "weight": weight,
        "density": density,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Example with density=True
    input_tensor = torch.tensor([1., 2, 1]).numpy()
    bins = 4
    range_val = (0., 3.)
    weight = torch.tensor([1., 2., 4.]).numpy()
    density = True
    out = None
    input_dict = {
        "input": input_tensor,
        "bins": bins,
        "range": range_val,
        "weight": weight,
        "density": density,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))


    # Input 4: Example with no weights
    input_tensor = torch.tensor([1., 2, 1]).numpy()
    bins = 4
    range_val = (0., 3.)
    weight = None
    density = False
    out = None
    input_dict = {
        "input": input_tensor,
        "bins": bins,
        "range": range_val,
        "weight": weight,
        "density": density,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Example with negative values and custom range
    input_tensor = torch.tensor([-1., 0, 1]).numpy()
    bins = 5
    range_val = (-2., 2.)
    weight = torch.tensor([1., 1., 1.]).numpy()
    density = False
    out = None
    input_dict = {
        "input": input_tensor,
        "bins": bins,
        "range": range_val,
        "weight": weight,
        "density": density,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.histogram_1"] = torch_histogram_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.histogram_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.histogram_1'.")

check_valid('torch.histogram', generated_inputs['torch.histogram_1'], lib="torch")
