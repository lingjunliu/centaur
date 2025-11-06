
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def rrelu_inputs():
    list_of_inputs = []

    # Input 1
    input_arr = np.array([-1.0, 0.0, 1.0, -0.5, 2.5], dtype=np.float32)
    input_dict = {
        "lower": 0.1,
        "upper": 0.3,
        "inplace": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_arr = np.array([[-2.0, -1.0, 0.0],
                          [1.0, 2.0, 3.0]], dtype=np.float32)
    input_dict = {
        "lower": 0.0,
        "upper": 0.1,
        "inplace": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_arr = np.array([[[ -1.5,  0.5,  2.0],
                           [ -0.3, -2.2,  1.1]],
                          [[  3.4, -4.5,  0.0],
                           [  2.2,  5.5, -0.7]]], dtype=np.float64)
    input_dict = {
        "lower": 0.05,
        "upper": 0.25,
        "inplace": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_arr = np.array([[[[ -1.0,  2.0],
                            [  3.0, -4.0]],
                           [[  0.0, -0.5],
                            [  1.5,  2.5]]]], dtype=np.float32)
    input_dict = {
        "lower": 0.2,
        "upper": 0.4,
        "inplace": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5 (scalar)
    input_arr = np.array(-3.5, dtype=np.float32)
    input_dict = {
        "lower": 0.25,
        "upper": 0.75,
        "inplace": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6 (large magnitude)
    input_arr = np.array([-1000.0, 1000.0, -500.0, 500.0], dtype=np.float32)
    input_dict = {
        "lower": 0.01,
        "upper": 0.02,
        "inplace": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 (all negative 2D)
    input_arr = np.array([[-1.0, -2.0, -3.0],
                          [-4.0, -5.0, -6.0],
                          [-7.0, -8.0, -9.0]], dtype=np.float32)
    input_dict = {
        "lower": 0.2,
        "upper": 0.4,
        "inplace": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8 (all positive 3D)
    input_arr = np.array([[[1.0, 2.0, 3.0, 4.0],
                           [5.0, 6.0, 7.0, 8.0],
                           [9.0, 10.0, 11.0, 12.0],
                           [13.0, 14.0, 15.0, 16.0]]], dtype=np.float32)
    input_dict = {
        "lower": 0.05,
        "upper": 0.15,
        "inplace": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 (non-contiguous via transpose)
    base = np.arange(24, dtype=np.float32).reshape(4, 6)
    input_arr = base.T
    input_dict = {
        "lower": 0.1,
        "upper": 0.2,
        "inplace": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 (5D tensor)
    input_arr = np.array([[[[[ -1.0,  0.0,  1.0],
                             [ -2.0,  2.0, -3.0]]]],
                          [[[[  4.0, -4.5,  5.5],
                             [  6.0, -6.5,  7.5]]]]], dtype=np.float64)
    input_dict = {
        "lower": 0.9,
        "upper": 0.99,
        "inplace": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11 (small slope range, near zero)
    input_arr = np.linspace(-1.0, 1.0, num=10, dtype=np.float32)
    input_dict = {
        "lower": 0.0001,
        "upper": 0.0002,
        "inplace": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12 (random mixture 4D)
    input_arr = np.array([[[[ -0.1,  0.2, -0.3],
                            [  0.4, -0.5,  0.6]],
                           [[ -0.7,  0.8, -0.9],
                            [  1.0, -1.1,  1.2]]]], dtype=np.float32)
    input_dict = {
        "lower": 0.15,
        "upper": 0.45,
        "inplace": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.RReLU"] = rrelu_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.RReLU' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.RReLU'.")


check_valid('torch.nn.RReLU', generated_inputs['torch.nn.RReLU'], lib="torch", suffix=0)
