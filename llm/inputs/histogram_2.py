
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def histogram_inputs():
    list_of_inputs = []

    # Example 1: Basic usage with int bins
    input = np.array([1., 2, 1])
    bins = 4
    range_val = (0., 3.)
    weight = np.array([1., 2., 4.])
    density = False
    out = None

    input_dict = {
        "input": input,
        "bins": bins,
        "range": range_val,
        "weight": weight,
        "density": density,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2: Using density=True
    input = np.array([1., 2, 1])
    bins = 4
    range_val = (0., 3.)
    weight = np.array([1., 2., 4.])
    density = True
    out = None

    input_dict = {
        "input": input,
        "bins": bins,
        "range": range_val,
        "weight": weight,
        "density": density,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 3: Using tensor bins
    input = np.array([1., 2, 1, 0.5, 2.5])
    bins = torch.tensor([0., 1., 2., 3.])
    range_val = None
    weight = None
    density = False
    out = None

    input_dict = {
        "input": input,
        "bins": bins,
        "range": range_val,
        "weight": weight,
        "density": density,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 4: Using negative values and different range
    input = np.array([-1., -2, -1, 0, 1])
    bins = 5
    range_val = (-3., 2.)
    weight = None
    density = False
    out = None

    input_dict = {
        "input": input,
        "bins": bins,
        "range": range_val,
        "weight": weight,
        "density": density,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
generated_inputs["torch.histogram_2"] = histogram_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.histogram_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.histogram_2'.")

check_valid('torch.histogram', generated_inputs['torch.histogram_2'], lib="torch")
