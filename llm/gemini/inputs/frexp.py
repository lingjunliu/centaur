
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def frexp_inputs():
    list_of_inputs = []

    # Input 1: 1D float tensor
    input1 = torch.arange(1, 9, dtype=torch.float32).numpy()
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D float tensor with negative values
    input2 = torch.randn(3, 4, dtype=torch.float64).numpy() * -1
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Scalar float tensor
    input3 = torch.tensor(3.14, dtype=torch.float32).numpy()
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Empty tensor
    input4 = torch.empty(0, dtype=torch.float32).numpy()
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 3D float tensor
    input5 = torch.randn(2, 3, 5, dtype=torch.float32).numpy()
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Input 6: 1D tensor with zeros
    input6 = torch.tensor([0.0, 1.0, 2.0, 0.0, 4.0], dtype=torch.float32).numpy()
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: Large values
    input7 = torch.tensor([2**30, 2**60, -2**30], dtype=torch.float64).numpy()
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = frexp_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('frexp', generated_inputs)
