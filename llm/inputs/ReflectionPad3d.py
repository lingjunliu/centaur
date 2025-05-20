
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def ReflectionPad3d_inputs():
    generated_inputs = []

    # Test case 1: int padding
    input1 = torch.arange(8, dtype=torch.float).reshape(1, 1, 2, 2, 2).numpy()
    padding1 = 1
    input_dict1 = {"padding": padding1, "input": input1}
    generated_inputs.append(copy.deepcopy(input_dict1))

    # Test case 2: tuple padding (different on each side)
    input2 = torch.randn(1, 3, 5, 5, 5).numpy()
    padding2 = (1, 2, 0, 1, 2, 0)
    input_dict2 = {"padding": padding2, "input": input2}
    generated_inputs.append(copy.deepcopy(input_dict2))

    # Test case 3: tuple padding (uniform)
    input3 = torch.randn(2, 4, 3, 3, 3).numpy()
    padding3 = (1, 1, 1, 1, 1, 1)
    input_dict3 = {"padding": padding3, "input": input3}
    generated_inputs.append(copy.deepcopy(input_dict3))

    # Test case 4: zero padding
    input4 = torch.randn(1, 1, 4, 4, 4).numpy()
    padding4 = 0
    input_dict4 = {"padding": padding4, "input": input4}
    generated_inputs.append(copy.deepcopy(input_dict4))

    # Test case 5: smaller padding
    input5 = torch.randn(1, 2, 3, 3, 3).numpy()
    padding5 = 1
    input_dict5 = {"padding": padding5, "input": input5}
    generated_inputs.append(copy.deepcopy(input_dict5))
    
    return generated_inputs

generated_inputs = ReflectionPad3d_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('ReflectionPad3d', generated_inputs)
