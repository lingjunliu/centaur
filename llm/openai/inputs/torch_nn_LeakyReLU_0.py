
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def leakyrelu_inputs():
    list_of_inputs = []

    # Input 1
    input_arr = torch.tensor([1.0, -2.0, 3.5, -4.2], dtype=torch.float32).numpy()
    negative_slope = 0.01
    inplace = False
    list_of_inputs.append(copy.deepcopy({
        "negative_slope": negative_slope,
        "inplace": inplace,
        "input": input_arr
    }))

    # Input 2
    input_arr = torch.randn(2, 3, dtype=torch.float32).numpy()
    negative_slope = 0.2
    inplace = True
    list_of_inputs.append(copy.deepcopy({
        "negative_slope": negative_slope,
        "inplace": inplace,
        "input": input_arr
    }))

    # Input 3
    input_arr = torch.linspace(-1, 1, steps=24, dtype=torch.float32).reshape(2, 3, 4).numpy()
    negative_slope = 1e-3
    inplace = False
    list_of_inputs.append(copy.deepcopy({
        "negative_slope": negative_slope,
        "inplace": inplace,
        "input": input_arr
    }))

    # Input 4
    input_arr = torch.randn(1, 3, 4, 4, dtype=torch.float64).numpy()
    negative_slope = 0.0
    inplace = False
    list_of_inputs.append(copy.deepcopy({
        "negative_slope": negative_slope,
        "inplace": inplace,
        "input": input_arr
    }))

    # Input 5
    input_arr = torch.tensor(3.14, dtype=torch.float32).numpy()
    negative_slope = 0.5
    inplace = True
    list_of_inputs.append(copy.deepcopy({
        "negative_slope": negative_slope,
        "inplace": inplace,
        "input": input_arr
    }))

    # Input 6
    input_arr = torch.empty(0, dtype=torch.float32).numpy()
    negative_slope = 0.1
    inplace = False
    list_of_inputs.append(copy.deepcopy({
        "negative_slope": negative_slope,
        "inplace": inplace,
        "input": input_arr
    }))

    # Input 7
    input_arr = torch.randn(1, 2, 3, 1, 4, dtype=torch.float32).numpy()
    negative_slope = 0.8
    inplace = True
    list_of_inputs.append(copy.deepcopy({
        "negative_slope": negative_slope,
        "inplace": inplace,
        "input": input_arr
    }))

    # Input 8
    input_arr = torch.tensor([float('nan'), float('inf'), -float('inf'), -1.0, 0.0, 2.0], dtype=torch.float32).numpy()
    negative_slope = 0.3
    inplace = False
    list_of_inputs.append(copy.deepcopy({
        "negative_slope": negative_slope,
        "inplace": inplace,
        "input": input_arr
    }))

    # Input 9
    input_arr = torch.randn(4, dtype=torch.float16).numpy()
    negative_slope = 1.5
    inplace = False
    list_of_inputs.append(copy.deepcopy({
        "negative_slope": negative_slope,
        "inplace": inplace,
        "input": input_arr
    }))

    # Input 10
    input_arr = torch.empty(2, 0, 3, dtype=torch.float32).numpy()
    negative_slope = 0.05
    inplace = True
    list_of_inputs.append(copy.deepcopy({
        "negative_slope": negative_slope,
        "inplace": inplace,
        "input": input_arr
    }))

    # Input 11
    input_arr = torch.linspace(-5, 5, steps=10, dtype=torch.float64).numpy()
    negative_slope = np.float32(0.07)
    inplace = np.bool_(False)
    list_of_inputs.append(copy.deepcopy({
        "negative_slope": negative_slope.item(),
        "inplace": bool(inplace),
        "input": input_arr
    }))

    # Input 12
    input_arr = torch.full((1, 1, 1, 1, 1, 1), -1.0, dtype=torch.float32).numpy()
    negative_slope = np.float64(2.0)
    inplace = np.bool_(True)
    list_of_inputs.append(copy.deepcopy({
        "negative_slope": float(negative_slope),
        "inplace": bool(inplace),
        "input": input_arr
    }))

    return list_of_inputs

generated_inputs["torch.nn.LeakyReLU"] = leakyrelu_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.LeakyReLU' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.LeakyReLU'.")


check_valid('torch.nn.LeakyReLU', generated_inputs['torch.nn.LeakyReLU'], lib="torch", suffix=0)
