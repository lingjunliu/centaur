
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def lu_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor, pivot=True
    A = torch.randn(3, 2).numpy()
    input_dict = {"A": A, "pivot": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float tensor, pivot=True
    A = torch.randn(2, 5).numpy()
    input_dict = {"A": A, "pivot": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Double tensor, pivot=True
    A = torch.randn(4, 4).double().numpy()
    input_dict = {"A": A, "pivot": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Complex tensor, pivot=True
    A = (torch.randn(2, 3) + 1j * torch.randn(2, 3)).numpy()
    input_dict = {"A": A, "pivot": True}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Complex tensor, pivot=True
    A = (torch.randn(3, 3) + 1j * torch.randn(3, 3)).numpy()
    input_dict = {"A": A, "pivot": True}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Batch of matrices, pivot=True
    A = torch.randn(2, 3, 4).numpy()
    input_dict = {"A": A, "pivot": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Batch of matrices, pivot=True
    A = torch.randn(3, 2, 2).numpy()
    input_dict = {"A": A, "pivot": True}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Rectangular matrix with negative values, pivot = True
    A = torch.randn(2, 4) * -1.0
    input_dict = {"A": A.numpy(), "pivot": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Rectangular matrix with negative values, pivot = True
    A = torch.randn(5, 3) * -1.0
    input_dict = {"A": A.numpy(), "pivot": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = lu_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('lu', generated_inputs)
