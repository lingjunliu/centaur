
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def threshold_inputs():
    list_of_inputs = []

    # Input 1
    input_arr = torch.arange(-3, 3, dtype=torch.float32).numpy()
    threshold = 0.0
    value = 0.5
    inplace = False
    input_dict = {
        "threshold": threshold,
        "value": value,
        "inplace": inplace,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_arr = torch.randn(2, 3, dtype=torch.float64).numpy()
    threshold = -1.5
    value = -2.0
    inplace = True
    input_dict = {
        "threshold": threshold,
        "value": value,
        "inplace": inplace,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_arr = torch.tensor([[1.0, 2.0, 3.0],
                              [-1.0, 0.0, 4.0]], dtype=torch.float32).numpy()
    threshold = 1.5
    value = 0.0
    inplace = False
    input_dict = {
        "threshold": threshold,
        "value": value,
        "inplace": inplace,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_arr = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).numpy()
    threshold = 10.0
    value = -10.0
    inplace = False
    input_dict = {
        "threshold": threshold,
        "value": value,
        "inplace": inplace,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5 (0-D tensor)
    input_arr = torch.tensor(0.0, dtype=torch.float32).numpy()
    threshold = 0.0
    value = 1.0
    inplace = True
    input_dict = {
        "threshold": threshold,
        "value": value,
        "inplace": inplace,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_arr = torch.zeros(5, dtype=torch.float64).numpy()
    threshold = -0.1
    value = -0.1
    inplace = True
    input_dict = {
        "threshold": threshold,
        "value": value,
        "inplace": inplace,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_arr = torch.linspace(-1, 1, 7, dtype=torch.float32).numpy()
    threshold = -0.5
    value = -0.5
    inplace = False
    input_dict = {
        "threshold": threshold,
        "value": value,
        "inplace": inplace,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8 (3D tensor)
    input_arr = torch.full((3, 3, 3), -2.0, dtype=torch.float32).numpy()
    threshold = -1.0
    value = 0.0
    inplace = True
    input_dict = {
        "threshold": threshold,
        "value": value,
        "inplace": inplace,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 (float16)
    input_arr = torch.randn(2, 2, 2, dtype=torch.float16).numpy()
    threshold = 0.1
    value = 0.0
    inplace = False
    input_dict = {
        "threshold": threshold,
        "value": value,
        "inplace": inplace,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 (4D tensor)
    input_arr = torch.rand(1, 4, 3, 2, dtype=torch.float32).numpy()
    threshold = 0.5
    value = 0.0
    inplace = True
    input_dict = {
        "threshold": threshold,
        "value": value,
        "inplace": inplace,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11 (empty tensor)
    input_arr = torch.tensor([], dtype=torch.float32).numpy()
    threshold = 0.0
    value = 0.0
    inplace = False
    input_dict = {
        "threshold": threshold,
        "value": value,
        "inplace": inplace,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12 (large magnitude)
    input_arr = torch.tensor([1e10, -1e10], dtype=torch.float64).numpy()
    threshold = 0.0
    value = 42.0
    inplace = False
    input_dict = {
        "threshold": threshold,
        "value": value,
        "inplace": inplace,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.Threshold"] = threshold_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.Threshold' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Threshold'.")


check_valid('torch.nn.Threshold', generated_inputs['torch.nn.Threshold'], lib="torch", suffix=0)
