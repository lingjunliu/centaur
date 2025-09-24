
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def torch_range_inputs():
    list_of_inputs = []

    # Input 1
    start = 1.0
    end = 5.0
    step = 1.0
    out = None
    dtype = None
    layout = "strided"
    requires_grad = False

    input_dict = {
        "start": np.float64(start),
        "end": np.float64(end),
        "step": np.float64(step),
        "out": None,
        "dtype": None,
        "layout": layout,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    start = 0.0
    end = 10.0
    step = 2.5
    out = None
    dtype = None
    layout = "strided"
    requires_grad = True

    input_dict = {
        "start":  np.float64(start),
        "end":  np.float64(end),
        "step":  np.float64(step),
        "out": None,
        "dtype": None,
        "layout": layout,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    start = -5.0
    end = 5.0
    step = 0.5
    out = None
    dtype = None
    layout = "strided"
    requires_grad = False

    input_dict = {
        "start":  np.float64(start),
        "end":  np.float64(end),
        "step":  np.float64(step),
        "out": None,
        "dtype": None,
        "layout": layout,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.range"] = torch_range_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.range' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.range'.")

check_valid('torch.range', generated_inputs['torch.range'], lib="torch", suffix=0)
