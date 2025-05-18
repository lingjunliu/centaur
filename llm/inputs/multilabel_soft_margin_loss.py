
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy
import numpy as np

def multilabel_soft_margin_loss_inputs():
    list_of_inputs = []

    input_tensor = torch.randn(3, 5)
    target_tensor = torch.randint(0, 2, (3, 5)).float()
    weight_tensor = torch.randn(5)
    input_dict = {
        "input": input_tensor.numpy(),
        "target": target_tensor.numpy(),
        "weight": weight_tensor.numpy(),
        "size_average": True,
        "reduce": True,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(2, 4)
    target_tensor = torch.randint(0, 2, (2, 4)).float()
    weight_tensor = None
    input_dict = {
        "input": input_tensor.numpy(),
        "target": target_tensor.numpy(),
        "weight": weight_tensor,
        "size_average": False,
        "reduce": True,
        "reduction": 'sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(5, 3)
    target_tensor = torch.randint(0, 2, (5, 3)).float()
    weight_tensor = torch.ones(3)
    input_dict = {
        "input": input_tensor.numpy(),
        "target": target_tensor.numpy(),
        "weight": weight_tensor.numpy(),
        "size_average": None,
        "reduce": False,
        "reduction": 'none'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(4, 2)
    target_tensor = torch.randint(0, 2, (4, 2)).float()
    weight_tensor = torch.tensor([0.5, 1.0])
    input_dict = {
        "input": input_tensor.numpy(),
        "target": target_tensor.numpy(),
        "weight": weight_tensor.numpy(),
        "size_average": True,
        "reduce": None,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(1, 6)
    target_tensor = torch.randint(0, 2, (1, 6)).float()
    weight_tensor = None
    input_dict = {
        "input": input_tensor.numpy(),
        "target": target_tensor.numpy(),
        "weight": weight_tensor,
        "size_average": False,
        "reduce": False,
        "reduction": 'sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

list_of_inputs = multilabel_soft_margin_loss_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('multilabel_soft_margin_loss', list_of_inputs)
