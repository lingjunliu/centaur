
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def index_copy_inputs():
    list_of_inputs = []

    # Case 1: Basic 1D tensor
    input_tensor = torch.randn(5).numpy()
    index_tensor = torch.tensor([0, 2, 4]).numpy()
    source_tensor = torch.randn(3).numpy()
    input_dict = {
        "input": input_tensor,
        "dim": 0,
        "index": index_tensor,
        "source": source_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: 2D tensor
    input_tensor = torch.randn(3, 4).numpy()
    index_tensor = torch.tensor([0, 2]).numpy()
    source_tensor = torch.randn(2, 4).numpy()
    input_dict = {
        "input": input_tensor,
        "dim": 0,
        "index": index_tensor,
        "source": source_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Different dim to copy along
    input_tensor = torch.randn(3, 4).numpy()
    index_tensor = torch.tensor([1, 3]).numpy()
    source_tensor = torch.randn(3, 2).numpy()
    input_dict = {
        "input": input_tensor,
        "dim": 1,
        "index": index_tensor,
        "source": source_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: 3D tensor
    input_tensor = torch.randn(2, 3, 4).numpy()
    index_tensor = torch.tensor([0, 1]).numpy()
    source_tensor = torch.randn(2, 3, 4).numpy()
    input_dict = {
        "input": input_tensor,
        "dim": 0,
        "index": index_tensor,
        "source": source_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Integer input tensor
    input_tensor = torch.randint(0, 10, (3, 4)).numpy()
    index_tensor = torch.tensor([0, 1]).numpy()
    source_tensor = torch.randint(0, 10, (2, 4)).numpy()
    input_dict = {
        "input": input_tensor,
        "dim": 0,
        "index": index_tensor,
        "source": source_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: Float input tensor, dim=1, different source shape to test exceptions
    input_tensor = torch.randn(5, 5).numpy()
    index_tensor = torch.tensor([0, 2, 4]).numpy()
    source_tensor = torch.randn(5, 3).numpy()

    input_dict = {
        "input": input_tensor,
        "dim": 1,
        "index": index_tensor,
        "source": source_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = index_copy_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('index_copy', generated_inputs)
