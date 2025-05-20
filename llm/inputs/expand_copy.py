
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def expand_copy_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D tensor expansion
    input1 = torch.tensor([1, 2, 3]).numpy()
    size1 = (3, 3)
    input_dict1 = {"input": input1, "size": size1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D tensor expansion
    input2 = torch.randn(2, 3).numpy()
    size2 = (2, 4, 3)
    input_dict2 = {"input": input2, "size": size2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Expanding a scalar to a higher dimension
    input3 = torch.tensor(5).numpy()
    size3 = (2, 3, 4)
    input_dict3 = {"input": input3, "size": size3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    return list_of_inputs

generated_inputs = expand_copy_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('expand_copy', generated_inputs)
