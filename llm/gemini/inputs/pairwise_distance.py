
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy
import numpy as np

def pairwise_distance_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensors
    x1 = torch.randn(10, 5).numpy()
    x2 = torch.randn(10, 5).numpy()
    input_dict = {
        "x1": x1,
        "x2": x2,
        "p": 2.0,
        "eps": 1e-6,
        "keepdim": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Integer tensors
    x1 = torch.randint(0, 10, (5, 3)).numpy()
    x2 = torch.randint(0, 10, (5, 3)).numpy()
    input_dict = {
        "x1": x1,
        "x2": x2,
        "p": 1.0,
        "eps": 1e-8,
        "keepdim": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different shapes (but still compatible)
    x1 = torch.randn(7, 4).numpy()
    x2 = torch.randn(7, 4).numpy()
    input_dict = {
        "x1": x1,
        "x2": x2,
        "p": 0.5,
        "eps": 1e-4,
        "keepdim": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative values, different p
    x1 = torch.randn(3, 2) * -1.0
    x2 = torch.randn(3, 2) * -1.0
    x1 = x1.numpy()
    x2 = x2.numpy()
    input_dict = {
        "x1": x1,
        "x2": x2,
        "p": 3.0,
        "eps": 1e-12,
        "keepdim": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different p value
    x1 = torch.randn(8, 6).numpy()
    x2 = torch.randn(8, 6).numpy()
    input_dict = {
        "x1": x1,
        "x2": x2,
        "p": 1.5,
        "eps": 1e-6,
        "keepdim": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Zero tensors
    x1 = torch.zeros(4, 2).numpy()
    x2 = torch.zeros(4, 2).numpy()
    input_dict = {
        "x1": x1,
        "x2": x2,
        "p": 2.0,
        "eps": 1e-6,
        "keepdim": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: One dimensional tensors
    x1 = torch.randn(5).numpy()
    x2 = torch.randn(5).numpy()
    input_dict = {
        "x1": x1,
        "x2": x2,
        "p": 2.0,
        "eps": 1e-6,
        "keepdim": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = pairwise_distance_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('pairwise_distance', generated_inputs)
