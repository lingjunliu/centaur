
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def narrow_copy_inputs():
    list_of_inputs = []

    # Case 1: Basic 2D tensor, positive start and length
    input_tensor = torch.randn(5, 5).numpy()
    input_dict = {
        "input": input_tensor,
        "dim": 0,
        "start": 1,
        "length": 3
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: 3D tensor, negative start, length covering end
    input_tensor = torch.arange(24).reshape(2, 3, 4).float().numpy()
    input_dict = {
        "input": input_tensor,
        "dim": 1,
        "start": -2,
        "length": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: 1D tensor, start at 0, full length
    input_tensor = torch.arange(10).int().numpy()
    input_dict = {
        "input": input_tensor,
        "dim": 0,
        "start": 0,
        "length": 10
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Complex tensor, specifying dim = 0
    input_tensor = torch.randn(3, 3, dtype=torch.complex64).numpy()
    input_dict = {
        "input": input_tensor,
        "dim": 0,
        "start": 0,
        "length": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: 4D tensor, narrowing along the last dimension
    input_tensor = torch.randn(2, 3, 4, 5).double().numpy()
    input_dict = {
        "input": input_tensor,
        "dim": 3,
        "start": 2,
        "length": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: Bool tensor
    input_tensor = torch.tensor([[True, False], [False, True]]).bool().numpy()
    input_dict = {
        "input": input_tensor,
        "dim": 1,
        "start": 0,
        "length": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = narrow_copy_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('narrow_copy', generated_inputs)
