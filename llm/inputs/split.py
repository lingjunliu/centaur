
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def split_inputs():
    list_of_inputs = []

    # Input 1: Basic case, split into equal chunks
    tensor = torch.randn(10).numpy()
    split_size_or_sections = 2
    dim = 0
    input_dict = {"tensor": tensor, "split_size_or_sections": split_size_or_sections, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Split into unequal chunks
    tensor = torch.randn(10).numpy()
    split_size_or_sections = [2, 3, 5]
    dim = 0
    input_dict = {"tensor": tensor, "split_size_or_sections": split_size_or_sections, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multi-dimensional tensor
    tensor = torch.randn(4, 4).numpy()
    split_size_or_sections = 2
    dim = 1
    input_dict = {"tensor": tensor, "split_size_or_sections": split_size_or_sections, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Splitting along different dimension
    tensor = torch.randn(2, 3, 4).numpy()
    split_size_or_sections = 2
    dim = 0
    input_dict = {"tensor": tensor, "split_size_or_sections": split_size_or_sections, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Splitting a 3D tensor into sections
    tensor = torch.randn(2, 6, 4).numpy()
    split_size_or_sections = [1, 2, 3]
    dim = 1
    input_dict = {"tensor": tensor, "split_size_or_sections": split_size_or_sections, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Splitting a tensor where split_size is larger than the dimension
    tensor = torch.randn(5).numpy()
    split_size_or_sections = 10
    dim = 0
    input_dict = {"tensor": tensor, "split_size_or_sections": split_size_or_sections, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Integer tensor
    tensor = torch.randint(0, 10, (5,)).numpy()
    split_size_or_sections = 2
    dim = 0
    input_dict = {"tensor": tensor, "split_size_or_sections": split_size_or_sections, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Empty tensor
    tensor = torch.empty(0).numpy()
    split_size_or_sections = 1
    dim = 0
    input_dict = {"tensor": tensor, "split_size_or_sections": split_size_or_sections, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = split_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('split', generated_inputs)
