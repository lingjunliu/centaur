
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def l1loss_inputs():
    list_of_inputs = []

    input_tensor = torch.randn(3, 5).numpy()
    target_tensor = torch.randn(3, 5).numpy()
    reduction_mode = 'mean'
    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "reduction": reduction_mode
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(2, 4, 6).numpy()
    target_tensor = torch.randn(2, 4, 6).numpy()
    reduction_mode = 'sum'
    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "reduction": reduction_mode
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(1, 10, 10).numpy()
    target_tensor = torch.randn(1, 10, 10).numpy()
    reduction_mode = 'mean'
    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "reduction": reduction_mode
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(5,).numpy()
    target_tensor = torch.randn(5,).numpy()
    reduction_mode = 'mean'
    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "reduction": reduction_mode
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(2, 2, 2, 2).numpy()
    target_tensor = torch.randn(2, 2, 2, 2).numpy()
    reduction_mode = 'mean'
    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "reduction": reduction_mode
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

list_of_inputs = l1loss_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('L1Loss', list_of_inputs)
