
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def randn_like_inputs():
    list_of_inputs = []

    # Input 1: Float tensor
    input1 = torch.randn(2, 3, 4).numpy()
    input_dict1 = {
        "input": input1,
        "dtype": None,
        "layout": None,
        "requires_grad": False,
        "memory_format": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 4: Specify dtype
    input4 = torch.randn(4, 4).numpy()
    input_dict4 = {
        "input": input4,
        "dtype": torch.float64,
        "layout": None,
        "requires_grad": False,
        "memory_format": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 1D tensor
    input5 = torch.randn(10).numpy()
    input_dict5 = {
        "input": input5,
        "dtype": None,
        "layout": None,
        "requires_grad": True,
        "memory_format": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Negative values
    input6 = torch.randn(3, 3) * -1.0
    input6 = input6.numpy()
    input_dict6 = {
        "input": input6,
        "dtype": None,
        "layout": None,
        "requires_grad": False,
        "memory_format": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Complex tensor
    input7 = torch.randn(2, 2, dtype=torch.complex64).numpy()
    input_dict7 = {
        "input": input7,
        "dtype": None,
        "layout": None,
        "requires_grad": False,
        "memory_format": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = randn_like_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('randn_like', generated_inputs)
