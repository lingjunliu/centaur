
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def cosine_similarity_inputs():
    list_of_inputs = []

    x1 = torch.randn(3, 5).numpy()
    x2 = torch.randn(3, 5).numpy()
    dim = 1
    eps = 1e-8
    input_dict = {
        "x1": x1,
        "x2": x2,
        "dim": dim,
        "eps": eps
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x1 = torch.randn(2, 4, 6).numpy()
    x2 = torch.randn(2, 4, 6).numpy()
    dim = 2
    eps = 1e-6
    input_dict = {
        "x1": x1,
        "x2": x2,
        "dim": dim,
        "eps": eps
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x1 = torch.randn(10).numpy()
    x2 = torch.randn(10).numpy()
    dim = 0
    eps = 1e-4
    input_dict = {
        "x1": x1,
        "x2": x2,
        "dim": dim,
        "eps": eps
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x1 = torch.randn(5, 5, 5).numpy()
    x2 = torch.randn(5, 5, 5).numpy()
    dim = 0
    eps = 1e-5
    input_dict = {
        "x1": x1,
        "x2": x2,
        "dim": dim,
        "eps": eps
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x1 = torch.randn(2, 3).numpy()
    x2 = torch.randn(2, 3).numpy()
    dim = 1
    eps = 1e-12
    input_dict = {
        "x1": x1,
        "x2": x2,
        "dim": dim,
        "eps": eps
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

list_of_inputs = cosine_similarity_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('cosine_similarity', list_of_inputs)
