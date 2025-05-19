
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def addbmm_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensors
    input1 = {
        'input': torch.randn(3, 2).numpy(),
        'batch1': torch.randn(3, 3, 4).numpy(),
        'batch2': torch.randn(3, 4, 2).numpy(),
        'beta': 1.0,
        'alpha': 1.0
    }
    list_of_inputs.append(copy.deepcopy(input1))

    # Input 2: Integer tensors
    input2 = {
        'input': torch.randint(0, 10, (2, 5), dtype=torch.int32).numpy(),
        'batch1': torch.randint(0, 10, (2, 2, 2), dtype=torch.int32).numpy(),
        'batch2': torch.randint(0, 10, (2, 2, 5), dtype=torch.int32).numpy(),
        'beta': 0.5,
        'alpha': 2.0
    }
    list_of_inputs.append(copy.deepcopy(input2))

    # Input 3: Different shaped tensors, beta=0
    input3 = {
        'input': torch.randn(2, 4).numpy(),
        'batch1': torch.randn(2, 3, 2).numpy(),
        'batch2': torch.randn(2, 2, 4).numpy(),
        'beta': 0.0,
        'alpha': 0.5
    }
    list_of_inputs.append(copy.deepcopy(input3))

    # Input 4: Negative values, different alpha
    input4 = {
        'input': torch.randn(5, 2).numpy(),
        'batch1': torch.randn(5, 4, 5).numpy(),
        'batch2': torch.randn(5, 5, 2).numpy(),
        'beta': 0.8,
        'alpha': -1.0
    }
    list_of_inputs.append(copy.deepcopy(input4))

    # Input 5: Larger tensors
    input5 = {
        'input': torch.randn(4, 3).numpy(),
        'batch1': torch.randn(4, 4, 4).numpy(),
        'batch2': torch.randn(4, 4, 3).numpy(),
        'beta': 1.0,
        'alpha': 1.0
    }
    list_of_inputs.append(copy.deepcopy(input5))
    
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
