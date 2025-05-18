
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def pairwise_distance_inputs():
    list_of_inputs = []

    x1 = torch.randn(10, 128).numpy()
    x2 = torch.randn(10, 128).numpy()
    p = 2.0
    eps = 1e-6
    keepdim = False
    input_dict = {
        "x1": x1,
        "x2": x2,
        "p": p,
        "eps": eps,
        "keepdim": keepdim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x1 = torch.randn(5, 64).numpy()
    x2 = torch.randn(5, 64).numpy()
    p = 1.5
    eps = 1e-8
    keepdim = True
    input_dict = {
        "x1": x1,
        "x2": x2,
        "p": p,
        "eps": eps,
        "keepdim": keepdim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x1 = torch.randn(20, 256).numpy()
    x2 = torch.randn(20, 256).numpy()
    p = 3.0
    eps = 1e-4
    keepdim = False
    input_dict = {
        "x1": x1,
        "x2": x2,
        "p": p,
        "eps": eps,
        "keepdim": keepdim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x1 = torch.randn(2, 32).numpy()
    x2 = torch.randn(2, 32).numpy()
    p = 2.5
    eps = 1e-5
    keepdim = True
    input_dict = {
        "x1": x1,
        "x2": x2,
        "p": p,
        "eps": eps,
        "keepdim": keepdim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x1 = torch.randn(8, 512).numpy()
    x2 = torch.randn(8, 512).numpy()
    p = 1.0
    eps = 1e-7
    keepdim = False
    input_dict = {
        "x1": x1,
        "x2": x2,
        "p": p,
        "eps": eps,
        "keepdim": keepdim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

list_of_inputs = pairwise_distance_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('PairwiseDistance', list_of_inputs)
