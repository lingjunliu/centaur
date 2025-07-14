
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_range_inputs():
    list_of_inputs = []

    # Input 1
    start = 1.0
    end = 4.0
    step = 1.0
    out = None
    dtype = None
    layout = "strided"
    requires_grad = False

    input_dict = {
        "start": np.float64(start),
        "end": np.float64(end),
        "step": np.float64(step),
        "out": out,
        "dtype": dtype,
        "layout": layout,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    start = 1.0
    end = 4.0
    step = 0.5
    out = None
    dtype = None
    layout = "strided"
    requires_grad = False

    input_dict = {
        "start": np.float64(start),
        "end": np.float64(end),
        "step": np.float64(step),
        "out": out,
        "dtype": dtype,
        "layout": layout,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    start = 0.0
    end = 5.0
    step = 1.5
    out = None
    dtype = np.float64
    layout = "strided"
    requires_grad = True

    input_dict = {
        "start": np.float64(start),
        "end": np.float64(end),
        "step": np.float64(step),
        "out": out,
        "dtype": dtype,
        "layout": layout,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    start = -2.0
    end = 2.0
    step = 0.75
    out = None
    dtype = np.float32
    layout = "strided"
    requires_grad = False

    input_dict = {
        "start": np.float64(start),
        "end": np.float64(end),
        "step": np.float64(step),
        "out": out,
        "dtype": dtype,
        "layout": layout,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    start = 10.0
    end = 1.0
    step = -1.0
    out = None
    dtype = None
    layout = "strided"
    requires_grad = True

    input_dict = {
        "start": np.float64(start),
        "end": np.float64(end),
        "step": np.float64(step),
        "out": out,
        "dtype": dtype,
        "layout": layout,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    start = 2.5
    end = 7.5
    step = 2.0
    out = None
    dtype = np.float64
    layout = "strided"
    requires_grad = False

    input_dict = {
        "start": np.float64(start),
        "end": np.float64(end),
        "step": np.float64(step),
        "out": out,
        "dtype": dtype,
        "layout": layout,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    start = -5.0
    end = 5.0
    step = 2.5
    out = None
    dtype = np.float32
    layout = "strided"
    requires_grad = True

    input_dict = {
        "start": np.float64(start),
        "end": np.float64(end),
        "step": np.float64(step),
        "out": out,
        "dtype": dtype,
        "layout": layout,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    start = 0.1
    end = 0.9
    step = 0.2
    out = None
    dtype = None
    layout = "strided"
    requires_grad = False

    input_dict = {
        "start": np.float64(start),
        "end": np.float64(end),
        "step": np.float64(step),
        "out": out,
        "dtype": dtype,
        "layout": layout,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 9
    start = -10.0
    end = -5.0
    step = 1.0
    out = None
    dtype = np.float64
    layout = "strided"
    requires_grad = True

    input_dict = {
        "start": np.float64(start),
        "end": np.float64(end),
        "step": np.float64(step),
        "out": out,
        "dtype": dtype,
        "layout": layout,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    start = 5.0
    end = -5.0
    step = -2.0
    out = None
    dtype = np.float32
    layout = "strided"
    requires_grad = False

    input_dict = {
        "start": np.float64(start),
        "end": np.float64(end),
        "step": np.float64(step),
        "out": out,
        "dtype": dtype,
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
