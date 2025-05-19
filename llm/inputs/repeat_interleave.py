
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def repeat_interleave_inputs():
    list_of_inputs = []

    # Test case 1: Basic 1D tensor and scalar repeats
    input_tensor = torch.tensor([1, 2, 3])
    repeats = 2
    input_dict = {"input": input_tensor, "repeats": repeats}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 2: 2D tensor and scalar repeats
    input_tensor = torch.tensor([[1, 2], [3, 4]])
    repeats = 3
    input_dict = {"input": input_tensor, "repeats": repeats}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 3: 1D tensor and tensor repeats
    input_tensor = torch.tensor([1, 2, 3])
    repeats = torch.tensor([1, 2, 3])
    input_dict = {"input": input_tensor, "repeats": repeats}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 4: 2D tensor and tensor repeats along axis 0
    input_tensor = torch.tensor([[1, 2], [3, 4], [5, 6]])
    repeats = torch.tensor([1, 2, 1])
    input_dict = {"input": input_tensor, "repeats": repeats}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 5: Float tensor
    input_tensor = torch.tensor([1.5, 2.5, 3.5])
    repeats = 2
    input_dict = {"input": input_tensor, "repeats": repeats}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Test case 6: scalar repeat with dimension
    input_tensor = torch.randn(2, 3, 4)
    repeats = 2
    input_dict = {"input": input_tensor, "repeats": repeats}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 7: Empty tensor
    input_tensor = torch.tensor([])
    repeats = 2
    input_dict = {"input": input_tensor, "repeats": repeats}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = repeat_interleave_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('repeat_interleave', generated_inputs)
