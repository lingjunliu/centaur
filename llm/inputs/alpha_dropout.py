
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy

def alpha_dropout_inputs():
    list_of_inputs = []

    input = torch.randn(2, 3, 4, 5).numpy()
    p = 0.5
    training = True
    inplace = False
    input_dict = {
        "input": input,
        "p": p,
        "training": training,
        "inplace": inplace
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 5, 5).numpy()
    p = 0.2
    training = False
    inplace = True
    input_dict = {
        "input": input,
        "p": p,
        "training": training,
        "inplace": inplace
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(3, 2).numpy()
    p = 0.8
    training = True
    inplace = False
    input_dict = {
        "input": input,
        "p": p,
        "training": training,
        "inplace": inplace
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(10).numpy()
    p = 0.1
    training = False
    inplace = True
    input_dict = {
        "input": input,
        "p": p,
        "training": training,
        "inplace": inplace
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(2, 2, 2, 2, 2).numpy()
    p = 0.3
    training = True
    inplace = False
    input_dict = {
        "input": input,
        "p": p,
        "training": training,
        "inplace": inplace
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

list_of_inputs = alpha_dropout_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('alpha_dropout', list_of_inputs)
