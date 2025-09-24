
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def fftshift_inputs():
    generated_inputs = []

    # Test case 1: 1D float tensor
    input_1 = torch.arange(-5, 5, dtype=torch.float32).numpy()
    input_dict_1 = {"input": input_1, "dim": None}
    generated_inputs.append(copy.deepcopy(input_dict_1))

    # Test case 2: 2D int tensor with specified dim
    input_2 = torch.arange(16).reshape(4, 4).to(torch.int32).numpy()
    input_dict_2 = {"input": input_2, "dim": (0,)}
    generated_inputs.append(copy.deepcopy(input_dict_2))

    # Test case 3: 3D complex tensor
    input_3 = torch.randn(2, 3, 4, dtype=torch.complex64).numpy()
    input_dict_3 = {"input": input_3, "dim": None}
    generated_inputs.append(copy.deepcopy(input_dict_3))

    # Test case 4: 2D float tensor with different dim
    input_4 = torch.randn(5, 5).numpy()
    input_dict_4 = {"input": input_4, "dim": (1,)}
    generated_inputs.append(copy.deepcopy(input_dict_4))

    # Test case 5: 1D tensor with only one element
    input_5 = torch.tensor([1.0]).numpy()
    input_dict_5 = {"input": input_5, "dim": None}
    generated_inputs.append(copy.deepcopy(input_dict_5))

    # Test case 6: 4D tensor with specified dimensions
    input_6 = torch.randn(2, 3, 4, 5).numpy()
    input_dict_6 = {"input": input_6, "dim": (0, 2)}
    generated_inputs.append(copy.deepcopy(input_dict_6))
    
    # Test case 7: 2D int tensor with negative values
    input_7 = torch.arange(-8, 8).reshape(4, 4).to(torch.int32).numpy()
    input_dict_7 = {"input": input_7, "dim": (0,1)}
    generated_inputs.append(copy.deepcopy(input_dict_7))

    return generated_inputs

generated_inputs = fftshift_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('fftshift', generated_inputs)
