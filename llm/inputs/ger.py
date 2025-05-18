
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def ger_inputs():
    list_of_inputs = []

    vec1 = torch.randn(5).numpy()
    vec2 = torch.randn(3).numpy()
    input_dict = {
        "vec1": vec1,
        "vec2": vec2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    vec1 = torch.randn(10).numpy()
    vec2 = torch.randn(1).numpy()
    input_dict = {
        "vec1": vec1,
        "vec2": vec2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    vec1 = torch.randn(1).numpy()
    vec2 = torch.randn(7).numpy()
    input_dict = {
        "vec1": vec1,
        "vec2": vec2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    vec1 = torch.randn(4).numpy()
    vec2 = torch.randn(5).numpy()
    input_dict = {
        "vec1": vec1,
        "vec2": vec2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    vec1 = torch.randn(4).numpy()
    vec2 = torch.randn(4).numpy()
    input_dict = {
        "vec1": vec1,
        "vec2": vec2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

list_of_inputs = ger_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('ger', list_of_inputs)
