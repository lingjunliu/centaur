
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def addbmm_inputs():
    list_of_inputs = []

    # Input 1: Basic case with float tensors
    input_dict = {
        "input": torch.randn(4, 2).numpy(),
        "batch1": torch.randn(3, 4, 5).numpy(),
        "batch2": torch.randn(3, 5, 2).numpy(),
        "beta": 1.0,
        "alpha": 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Integer tensors
    input_dict = {
        "input": torch.randint(0, 10, (4, 5)).numpy(),
        "batch1": torch.randint(0, 10, (2, 4, 3)).numpy(),
        "batch2": torch.randint(0, 10, (2, 3, 5)).numpy(),
        "beta": 2.0,
        "alpha": 0.5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different beta and alpha, negative values
    input_dict = {
        "input": torch.randn(5, 3).numpy(),
        "batch1": torch.randn(4, 5, 6).numpy(),
        "batch2": torch.randn(4, 6, 3).numpy(),
        "beta": -0.5,
        "alpha": -2.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Larger dimensions
    input_dict = {
        "input": torch.randn(6, 4).numpy(),
        "batch1": torch.randn(5, 6, 7).numpy(),
        "batch2": torch.randn(5, 7, 4).numpy(),
        "beta": 0.0,
        "alpha": 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: beta = 0
    input_dict = {
        "input": torch.randn(4, 2).numpy(),
        "batch1": torch.randn(3, 4, 5).numpy(),
        "batch2": torch.randn(3, 5, 2).numpy(),
        "beta": 0.0,
        "alpha": 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = addbmm_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('addbmm', generated_inputs)
