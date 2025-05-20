
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def swapdims_inputs():
    list_of_inputs = []

    # Input 1: 3D float tensor
    input_tensor = torch.randn(2, 3, 4).numpy()
    input_dict = {"input": input_tensor, "dim0": 0, "dim1": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 4D int tensor
    input_tensor = torch.randint(0, 10, (2, 2, 3, 3)).numpy()
    input_dict = {"input": input_tensor, "dim0": 1, "dim1": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D complex tensor
    input_tensor = torch.randn(2, 3, dtype=torch.complex64).numpy()
    input_dict = {"input": input_tensor, "dim0": 0, "dim1": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 5D tensor, negative dimension
    input_tensor = torch.randn(2, 3, 4, 5, 6).numpy()
    input_dict = {"input": input_tensor, "dim0": 0, "dim1": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D tensor
    input_tensor = torch.randn(5).numpy()
    input_dict = {"input": input_tensor, "dim0": 0, "dim1": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D tensor, same dimensions
    input_tensor = torch.randn(2, 3, 4).numpy()
    input_dict = {"input": input_tensor, "dim0": 1, "dim1": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D tensor, different datatypes
    input_tensor = torch.rand(2, 2, 3, 3).double().numpy()
    input_dict = {"input": input_tensor, "dim0": 1, "dim1": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = swapdims_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('swapdims', generated_inputs)
