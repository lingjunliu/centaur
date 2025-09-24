
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import torch.nn as nn
import numpy as np
import copy

def mish_inputs():
    list_of_inputs = []

    # Input 1: 1D tensor, float
    input1 = torch.randn(5).numpy()
    input_dict1 = {"input": input1, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D tensor, float, negative values
    input2 = torch.randn(2, 3) * -1.0
    input2 = input2.numpy()
    input_dict2 = {"input": input2, "inplace": True}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D tensor, float
    input3 = torch.randn(2, 3, 4).numpy()
    input_dict3 = {"input": input3, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 4D tensor, float
    input4 = torch.randn(2, 3, 4, 5).numpy()
    input_dict4 = {"input": input4, "inplace": True}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 5D tensor, float
    input5 = torch.randn(2, 3, 4, 5, 6).numpy()
    input_dict5 = {"input": input5, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Scalar float
    input7 = torch.randn(1).item()
    input_dict7 = {"input": np.array(input7), "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs

generated_inputs = mish_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('Mish', generated_inputs)
