
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def pairwise_distance_inputs():
    list_of_inputs = []

    x1 = np.random.randn(10, 5).astype(np.float32)
    x2 = np.random.randn(10, 5).astype(np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "p": 2.0,
        "eps": 1e-6,
        "keepdim": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x1 = np.random.randn(5, 3, 2).astype(np.float64)
    x2 = np.random.randn(5, 3, 2).astype(np.float64)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "p": 1.5,
        "eps": 1e-8,
        "keepdim": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x1 = np.random.randn(20, 1).astype(np.float32)
    x2 = np.random.randn(20, 1).astype(np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "p": 0.0,
        "eps": 1e-4,
        "keepdim": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x1 = np.random.randn(3, 4, 5).astype(np.float32)
    x2 = np.random.randn(3, 4, 5).astype(np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "p": 3.0,
        "eps": 0.0,
        "keepdim": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x1 = np.random.randn(7, 2, 3, 4).astype(np.float32)
    x2 = np.random.randn(7, 2, 3, 4).astype(np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "p": 1.0,
        "eps": 1e-12,
        "keepdim": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x1 = np.random.randn(4, 2).astype(np.float32)
    x2 = np.random.randn(4, 2).astype(np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "p": 2.0,
        "eps": 1e-6,
        "keepdim": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x1 = np.random.randn(8, 3).astype(np.float32)
    x2 = np.random.randn(8, 3).astype(np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "p": 2.5,
        "eps": 1e-5,
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

check_valid('PairwiseDistance', generated_inputs)
