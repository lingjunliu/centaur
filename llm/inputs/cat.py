
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def cat_inputs():
    list_of_inputs = []

    # Example 1
    tensors = [torch.randn(2, 3).numpy(), torch.randn(2, 3).numpy()]
    dim = 0
    input_dict = {"tensors": tensors, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2
    tensors = [torch.randn(2, 3, 4).numpy(), torch.randn(2, 3, 4).numpy(), torch.randn(2, 3, 4).numpy()]
    dim = 1
    input_dict = {"tensors": tensors, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 3
    tensors = [torch.randn(1, 5).numpy(), torch.randn(1, 5).numpy(), torch.randn(1, 5).numpy(), torch.randn(1, 5).numpy()]
    dim = 1
    input_dict = {"tensors": tensors, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Example 4
    tensors = [torch.randn(3, 1, 2).numpy(), torch.randn(3, 1, 2).numpy()]
    dim = 2
    input_dict = {"tensors": tensors, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 5
    tensors = [torch.randn(4, 2, 3).numpy(), torch.randn(4, 2, 3).numpy()]
    dim = 0
    input_dict = {"tensors": tensors, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

list_of_inputs = cat_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('cat', list_of_inputs)
