
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def bilinear_inputs():
    list_of_inputs = []

    # Input 1
    input1 = np.random.rand(2, 3, 4).astype(np.float32)
    input2 = np.random.rand(2, 3, 5).astype(np.float32)
    weight = np.random.rand(6, 4, 5).astype(np.float32)
    bias = np.random.rand(6).astype(np.float32)

    input_dict = {
        "input1": input1,
        "input2": input2,
        "weight": weight,
        "bias": bias
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input1 = np.random.rand(1, 2, 3).astype(np.float32)
    input2 = np.random.rand(1, 2, 4).astype(np.float32)
    weight = np.random.rand(5, 3, 4).astype(np.float32)
    bias = np.random.rand(5).astype(np.float32)

    input_dict = {
        "input1": input1,
        "input2": input2,
        "weight": weight,
        "bias": bias
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input1 = np.random.rand(3, 1, 2).astype(np.float32)
    input2 = np.random.rand(3, 1, 3).astype(np.float32)
    weight = np.random.rand(4, 2, 3).astype(np.float32)
    bias = np.random.rand(4).astype(np.float32)

    input_dict = {
        "input1": input1,
        "input2": input2,
        "weight": weight,
        "bias": bias
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4
    input1 = np.random.rand(1, 5, 6).astype(np.float32)
    input2 = np.random.rand(1, 5, 7).astype(np.float32)
    weight = np.random.rand(8, 6, 7).astype(np.float32)
    bias = np.random.rand(8).astype(np.float32)

    input_dict = {
        "input1": input1,
        "input2": input2,
        "weight": weight,
        "bias": bias
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input1 = np.random.rand(2, 4).astype(np.float32)
    input2 = np.random.rand(2, 5).astype(np.float32)
    weight = np.random.rand(3, 4, 5).astype(np.float32)
    bias = np.random.rand(3).astype(np.float32)

    input_dict = {
        "input1": input1,
        "input2": input2,
        "weight": weight,
        "bias": bias
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6 (negative values)
    input1 = np.random.randn(2, 3, 4).astype(np.float32)
    input2 = np.random.randn(2, 3, 5).astype(np.float32)
    weight = np.random.randn(6, 4, 5).astype(np.float32)
    bias = np.random.randn(6).astype(np.float32)

    input_dict = {
        "input1": input1,
        "input2": input2,
        "weight": weight,
        "bias": bias
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 (different batch size)
    input1 = np.random.randn(5, 3, 4).astype(np.float32)
    input2 = np.random.randn(5, 3, 5).astype(np.float32)
    weight = np.random.randn(6, 4, 5).astype(np.float32)
    bias = np.random.randn(6).astype(np.float32)

    input_dict = {
        "input1": input1,
        "input2": input2,
        "weight": weight,
        "bias": bias
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input1 = np.random.rand(1, 1, 2).astype(np.float32)
    input2 = np.random.rand(1, 1, 3).astype(np.float32)
    weight = np.random.rand(4, 2, 3).astype(np.float32)
    bias = np.random.rand(4).astype(np.float32)

    input_dict = {
        "input1": input1,
        "input2": input2,
        "weight": weight,
        "bias": bias
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    input1 = np.random.rand(3, 5, 6).astype(np.float32)
    input2 = np.random.rand(3, 5, 7).astype(np.float32)
    weight = np.random.rand(8, 6, 7).astype(np.float32)
    bias = np.random.rand(8).astype(np.float32)

    input_dict = {
        "input1": input1,
        "input2": input2,
        "weight": weight,
        "bias": bias
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 (no bias)
    input1 = np.random.randn(2, 3, 4).astype(np.float32)
    input2 = np.random.randn(2, 3, 5).astype(np.float32)
    weight = np.random.randn(6, 4, 5).astype(np.float32)
    bias = None

    input_dict = {
        "input1": input1,
        "input2": input2,
        "weight": weight,
        "bias": bias
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11 (no batch size)
    input1 = np.random.randn(3).astype(np.float32)
    input2 = np.random.randn(5).astype(np.float32)
    weight = np.random.randn(6, 3, 5).astype(np.float32)
    bias = np.random.randn(6).astype(np.float32)

    input_dict = {
        "input1": input1,
        "input2": input2,
        "weight": weight,
        "bias": bias
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.bilinear"] = bilinear_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.bilinear' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.bilinear'.")

check_valid('torch.nn.functional.bilinear', generated_inputs['torch.nn.functional.bilinear'], lib="torch", suffix=0)
