
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def l1_loss_inputs():
    list_of_inputs = []

    input1 = torch.randn(3, 5).numpy()
    target1 = torch.randn(3, 5).numpy()
    input_dict1 = {"input": input1, "target": target1, "reduction": 'mean'}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(2, 4, 6).numpy()
    target2 = torch.randn(2, 4, 6).numpy()
    input_dict2 = {"input": input2, "target": target2, "reduction": 'sum'}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input4 = torch.randn(10).numpy()
    target4 = torch.randn(10).numpy()
    input_dict4 = {"input": input4, "target": target4, "reduction": 'mean'}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(2, 2, 2, 2).numpy()
    target5 = torch.randn(2, 2, 2, 2).numpy()
    input_dict5 = {"input": input5, "target": target5, "reduction": 'sum'}
    list_of_inputs.append(copy.deepcopy(input_dict5))


    return list_of_inputs

generated_inputs = l1_loss_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('l1_loss', generated_inputs)
