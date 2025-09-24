
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def bilinear_inputs():
    list_of_inputs = []

    # Input 1
    in1_features = 20
    in2_features = 30
    out_features = 40
    bias = True
    dtype = np.float32
    input1 = np.random.randn(128, in1_features).astype(dtype)
    input2 = np.random.randn(128, in2_features).astype(dtype)
    input_dict = {
        "in1_features": in1_features,
        "in2_features": in2_features,
        "out_features": out_features,
        "bias": bias,
        "dtype": torch.float32,
        "input1": input1,
        "input2": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    in1_features = 10
    in2_features = 15
    out_features = 20
    bias = False
    dtype = np.float64
    input1 = np.random.randn(64, in1_features).astype(dtype)
    input2 = np.random.randn(64, in2_features).astype(dtype)
    input_dict = {
        "in1_features": in1_features,
        "in2_features": in2_features,
        "out_features": out_features,
        "bias": bias,
        "dtype": torch.float64,
        "input1": input1,
        "input2": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    in1_features = 5
    in2_features = 7
    out_features = 3
    bias = True
    dtype = np.float32
    input1 = np.random.randn(32, 1, in1_features).astype(dtype)
    input2 = np.random.randn(32, 1, in2_features).astype(dtype)
    input_dict = {
        "in1_features": in1_features,
        "in2_features": in2_features,
        "out_features": out_features,
        "bias": bias,
        "dtype": torch.float32,
        "input1": input1,
        "input2": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    in1_features = 12
    in2_features = 18
    out_features = 24
    bias = False
    dtype = np.float64
    input1 = np.random.randn(1, 128, in1_features).astype(dtype)
    input2 = np.random.randn(1, 128, in2_features).astype(dtype)
    input_dict = {
        "in1_features": in1_features,
        "in2_features": in2_features,
        "out_features": out_features,
        "bias": bias,
        "dtype": torch.float64,
        "input1": input1,
        "input2": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    in1_features = 8
    in2_features = 11
    out_features = 15
    bias = True
    dtype = np.float32
    input1 = np.random.randn(2, 3, 4, in1_features).astype(dtype)
    input2 = np.random.randn(2, 3, 4, in2_features).astype(dtype)
    input_dict = {
        "in1_features": in1_features,
        "in2_features": in2_features,
        "out_features": out_features,
        "bias": bias,
        "dtype": torch.float32,
        "input1": input1,
        "input2": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    in1_features = 3
    in2_features = 5
    out_features = 2
    bias = False
    dtype = np.float64
    input1 = np.random.randn(10, in1_features).astype(dtype)
    input2 = np.random.randn(10, in2_features).astype(dtype)
    input_dict = {
        "in1_features": in1_features,
        "in2_features": in2_features,
        "out_features": out_features,
        "bias": bias,
        "dtype": torch.float64,
        "input1": input1,
        "input2": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    in1_features = 7
    in2_features = 9
    out_features = 11
    bias = True
    dtype = np.float32
    input1 = np.random.randn(5, 2, in1_features).astype(dtype)
    input2 = np.random.randn(5, 2, in2_features).astype(dtype)
    input_dict = {
        "in1_features": in1_features,
        "in2_features": in2_features,
        "out_features": out_features,
        "bias": bias,
        "dtype": torch.float32,
        "input1": input1,
        "input2": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    in1_features = 4
    in2_features = 6
    out_features = 8
    bias = False
    dtype = np.float64
    input1 = np.random.randn(2, 1, 3, in1_features).astype(dtype)
    input2 = np.random.randn(2, 1, 3, in2_features).astype(dtype)
    input_dict = {
        "in1_features": in1_features,
        "in2_features": in2_features,
        "out_features": out_features,
        "bias": bias,
        "dtype": torch.float64,
        "input1": input1,
        "input2": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    in1_features = 6
    in2_features = 8
    out_features = 10
    bias = True
    dtype = np.float32
    input1 = np.random.randn(3, 4, in1_features).astype(dtype)
    input2 = np.random.randn(3, 4, in2_features).astype(dtype)
    input_dict = {
        "in1_features": in1_features,
        "in2_features": in2_features,
        "out_features": out_features,
        "bias": bias,
        "dtype": torch.float32,
        "input1": input1,
        "input2": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    in1_features = 9
    in2_features = 12
    out_features = 15
    bias = False
    dtype = np.float64
    input1 = np.random.randn(4, 2, 3, in1_features).astype(dtype)
    input2 = np.random.randn(4, 2, 3, in2_features).astype(dtype)
    input_dict = {
        "in1_features": in1_features,
        "in2_features": in2_features,
        "out_features": out_features,
        "bias": bias,
        "dtype": torch.float64,
        "input1": input1,
        "input2": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.Bilinear"] = bilinear_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.Bilinear' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Bilinear'.")

check_valid('torch.nn.Bilinear', generated_inputs['torch.nn.Bilinear'], lib="torch", suffix=0)
