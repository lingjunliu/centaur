
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def layer_norm_inputs():
    list_of_inputs = []

    input = torch.randn(2, 3, 4).numpy()
    normalized_shape = [4]
    eps = 1e-5
    elementwise_affine = True
    input_dict = {
        "input": input,
        "normalized_shape": normalized_shape,
        "eps": eps,
        "elementwise_affine": elementwise_affine
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 5, 5, 2).numpy()
    normalized_shape = [2]
    eps = 1e-8
    elementwise_affine = False
    input_dict = {
        "input": input,
        "normalized_shape": normalized_shape,
        "eps": eps,
        "elementwise_affine": elementwise_affine
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(3, 10).numpy()
    normalized_shape = [10]
    eps = 1e-12
    elementwise_affine = True
    input_dict = {
        "input": input,
        "normalized_shape": normalized_shape,
        "eps": eps,
        "elementwise_affine": elementwise_affine
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(4, 2, 3, 5).numpy()
    normalized_shape = [3, 5]
    eps = 1e-6
    elementwise_affine = False
    input_dict = {
        "input": input,
        "normalized_shape": normalized_shape,
        "eps": eps,
        "elementwise_affine": elementwise_affine
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 8, 8).numpy()
    normalized_shape = [8, 8]
    eps = 1e-3
    elementwise_affine = True
    input_dict = {
        "input": input,
        "normalized_shape": normalized_shape,
        "eps": eps,
        "elementwise_affine": elementwise_affine
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

list_of_inputs = layer_norm_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('layer_norm', list_of_inputs)
