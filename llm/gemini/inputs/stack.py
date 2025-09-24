
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def stack_inputs():
    list_of_inputs = []

    # Input 1: Basic example with float tensors and dim=0
    x = torch.randn(2, 3).numpy()
    tensors = (x, x)
    dim = 0
    input_dict = {"tensors": tensors, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Int tensors with dim=1
    x = torch.randint(0, 10, (3, 4)).numpy()
    tensors = (x, x, x)
    dim = 1
    input_dict = {"tensors": tensors, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Three-dimensional tensors with dim=2
    x = torch.randn(2, 3, 5).numpy()
    tensors = (x, x)
    dim = 2
    input_dict = {"tensors": tensors, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative dim (equivalent to dim=2 in this case)
    x = torch.randn(4, 2).numpy()
    tensors = (x, x)
    dim = -1
    input_dict = {"tensors": tensors, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: One-dimensional tensors with dim=0
    x = torch.randn(5).numpy()
    tensors = (x, x, x, x)
    dim = 0
    input_dict = {"tensors": tensors, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Zero-dimensional tensor
    x = torch.tensor(5.0).numpy()
    tensors = (x, x)
    dim = 0
    input_dict = {"tensors": tensors, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = stack_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('stack', generated_inputs)
