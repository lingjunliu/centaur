
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def poisson_nll_loss_inputs():
    list_of_inputs = []

    input_dict = {
        "input": torch.randn(3, 5).numpy(),
        "target": torch.randint(0, 10, (3, 5)).float().numpy(),
        "log_input": False,
        "full": False,
        "eps": 1e-8,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": torch.randn(2, 4, 3).numpy(),
        "target": torch.randint(0, 5, (2, 4, 3)).float().numpy(),
        "log_input": True,
        "full": True,
        "eps": 1e-6,
        "reduction": 'sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": torch.randn(1, 7).numpy(),
        "target": torch.randint(0, 15, (1, 7)).float().numpy(),
        "log_input": False,
        "full": True,
        "eps": 1e-10,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": torch.randn(4, 2, 2).numpy(),
        "target": torch.randint(0, 8, (4, 2, 2)).float().numpy(),
        "log_input": True,
        "full": False,
        "eps": 1e-4,
        "reduction": 'sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": torch.randn(2, 3, 4, 5).numpy(),
        "target": torch.randint(0, 3, (2, 3, 4, 5)).float().numpy(),
        "log_input": False,
        "full": False,
        "eps": 1e-5,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

list_of_inputs = poisson_nll_loss_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('PoissonNLLLoss', list_of_inputs)
