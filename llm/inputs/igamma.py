
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def igamma_inputs():
    list_of_inputs = []

    input1 = torch.randn(2, 3).abs().numpy()
    other1 = torch.randn(2, 3).abs().numpy()
    input_dict1 = {"input": input1, "other": other1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(5).abs().numpy()
    other2 = torch.randn(5).abs().numpy()
    input_dict2 = {"input": input2, "other": other2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(1, 4, 4).abs().numpy()
    other3 = torch.randn(1, 4, 4).abs().numpy()
    input_dict3 = {"input": input3, "other": other3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = (torch.rand(2, 2) * 10).numpy()
    other4 = (torch.rand(2, 2) * 10).numpy()
    input_dict4 = {"input": input4, "other": other4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = (torch.rand(3, 1, 5) * 5).numpy()
    other5 = (torch.rand(3, 1, 5) * 5).numpy()
    input_dict5 = {"input": input5, "other": other5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

list_of_inputs = igamma_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('igamma', list_of_inputs)
