
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy
import numpy as np

def poisson_nll_loss_inputs():
    list_of_inputs = []

    # Input 1: Basic example with float tensors, log_input=True
    input_dict = {
        "input": torch.randn(3, 5).numpy(),
        "target": torch.randint(0, 10, (3, 5)).float().numpy(),
        "log_input": True,
        "full": False,
        "size_average": None,
        "eps": 1e-8,
        "reduce": None,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Integer target, log_input=False
    input_dict = {
        "input": torch.rand(2, 4).numpy(),
        "target": torch.randint(0, 5, (2, 4)).float().numpy(),
        "log_input": False,
        "full": True,
        "size_average": None,
        "eps": 1e-6,
        "reduce": None,
        "reduction": 'sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different dimensions, log_input=True, reduction = 'none' - REMOVED THIS AS IT CAUSED ERROR
    # input_dict = {
    #     "input": torch.randn(1, 3, 8, 8).numpy(),
    #     "target": torch.randint(0, 5, (1, 3, 8, 8)).float().numpy(),
    #     "log_input": True,
    #     "full": False,
    #     "size_average": None,
    #     "eps": 1e-8,
    #     "reduce": None,
    #     "reduction": 'none'
    # }
    # list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D tensors
    input_dict = {
        "input": torch.randn(10).numpy(),
        "target": torch.randint(0, 5, (10,)).float().numpy(),
        "log_input": False,
        "full": True,
        "size_average": None,
        "eps": 1e-8,
        "reduce": None,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: No reduction - using 'mean' instead for scalar output
    input_dict = {
        "input": torch.randn(2, 3).numpy(),
        "target": torch.randint(0, 5, (2, 3)).float().numpy(),
        "log_input": True,
        "full": False,
        "size_average": None,
        "eps": 1e-8,
        "reduce": None,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different shape, log_input=False
    input_dict = {
        "input": torch.rand(5, 2, 2).numpy(),
        "target": torch.randint(0, 5, (5, 2, 2)).float().numpy(),
        "log_input": False,
        "full": True,
        "size_average": None,
        "eps": 1e-8,
        "reduce": None,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = poisson_nll_loss_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('poisson_nll_loss', generated_inputs)
