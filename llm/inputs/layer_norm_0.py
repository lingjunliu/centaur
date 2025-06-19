
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def layer_norm_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = torch.randn(2, 3, 4).numpy()
    normalized_shape = (3, 4)
    weight = torch.randn(4).numpy()
    bias = torch.randn(4).numpy()
    eps = 1e-5
    elementwise_affine = True

    input_dict = {
        "input": input_tensor,
        "normalized_shape": normalized_shape,
        "weight": weight,
        "bias": bias,
        "eps": eps,
        "elementwise_affine": elementwise_affine
    }

    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 2
    input_tensor = torch.randn(5, 5).numpy()
    normalized_shape = (5,)
    weight = torch.randn(5).numpy()
    bias = torch.randn(5).numpy()
    eps = 1e-8
    elementwise_affine = True

    input_dict = {
        "input": input_tensor,
        "normalized_shape": normalized_shape,
        "weight": weight,
        "bias": bias,
        "eps": eps,
        "elementwise_affine": elementwise_affine
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = torch.randn(1, 2, 3, 4).numpy()
    normalized_shape = (2, 3, 4)
    weight = torch.randn(4).numpy()
    bias = torch.randn(4).numpy()
    eps = 1e-12
    elementwise_affine = True
    input_dict = {
        "input": input_tensor,
        "normalized_shape": normalized_shape,
        "weight": weight,
        "bias": bias,
        "eps": eps,
        "elementwise_affine": elementwise_affine
    }

    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = torch.randn(2, 2).numpy()
    normalized_shape = (2,)
    weight = torch.randn(2).numpy()
    bias = torch.randn(2).numpy()
    eps = 1e-3
    elementwise_affine = True

    input_dict = {
        "input": input_tensor,
        "normalized_shape": normalized_shape,
        "weight": weight,
        "bias": bias,
        "eps": eps,
        "elementwise_affine": elementwise_affine
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = torch.randn(3, 10).numpy()
    normalized_shape = (10,)
    weight = torch.randn(10).numpy()
    bias = torch.randn(10).numpy()
    eps = 1e-7
    elementwise_affine = False

    input_dict = {
        "input": input_tensor,
        "normalized_shape": normalized_shape,
        "weight": weight,
        "bias": bias,
        "eps": eps,
        "elementwise_affine": elementwise_affine
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.layer_norm"] = layer_norm_inputs()


def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.layer_norm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.layer_norm'.")

check_valid('torch.nn.functional.layer_norm', generated_inputs['torch.nn.functional.layer_norm'], lib="torch", suffix=0)
