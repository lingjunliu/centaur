
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def swapaxes_inputs():
    list_of_inputs = []

    # Example 1: 3D tensor, swapping axes 0 and 1
    x = torch.tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]]]).numpy()
    input_dict = {"input": x, "axis0": 0, "axis1": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2: 3D tensor, swapping axes 0 and 2
    x = torch.tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]]]).numpy()
    input_dict = {"input": x, "axis0": 0, "axis1": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 3: 2D tensor, swapping axes 0 and 1 (same as transpose)
    x = torch.randn(2, 3).numpy()
    input_dict = {"input": x, "axis0": 0, "axis1": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 4: 4D tensor, swapping axes 1 and 3
    x = torch.randn(2, 3, 4, 5).numpy()
    input_dict = {"input": x, "axis0": 1, "axis1": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 5: 1D tensor, swapping axes 0 and 0 (no change)
    x = torch.arange(5).numpy()
    input_dict = {"input": x, "axis0": 0, "axis1": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Example 6: 3D integer tensor
    x = torch.randint(0, 10, (2, 3, 4)).numpy()
    input_dict = {"input": x, "axis0": 0, "axis1": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 7: 4D complex tensor
    x = torch.randn(2, 3, 4, 5, dtype=torch.complex64).numpy()
    input_dict = {"input": x, "axis0": 2, "axis1": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = swapaxes_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('swapaxes', generated_inputs)
