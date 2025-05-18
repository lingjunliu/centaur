
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def MSELoss_inputs():
    list_of_inputs = []

    input_tensor = torch.randn(3, 5).numpy()
    target_tensor = torch.randn(3, 5).numpy()
    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "size_average": True,
        "reduce": True,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(2, 4, 3).numpy()
    target_tensor = torch.randn(2, 4, 3).numpy()
    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "size_average": False,
        "reduce": True,
        "reduction": 'sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(1, 10).numpy()
    target_tensor = torch.randn(1, 10).numpy()
    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "size_average": None,
        "reduce": None,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(4, 4).numpy()
    target_tensor = torch.randn(4, 4).numpy()
    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "size_average": None,
        "reduce": None,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(2, 3, 2, 2).numpy()
    target_tensor = torch.randn(2, 3, 2, 2).numpy()
    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "size_average": None,
        "reduce": None,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

list_of_inputs = MSELoss_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('MSELoss', list_of_inputs)
