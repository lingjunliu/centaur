
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def index_select_inputs():
    list_of_inputs = []

    # Case 1: 2D float tensor, dim=0, integer index
    input_tensor = torch.randn(3, 4).numpy()
    index_tensor = torch.tensor([0, 2]).numpy()
    input_dict = {"input": input_tensor, "dim": 0, "index": index_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: 3D int tensor, dim=1, integer index with negative values
    input_tensor = torch.randint(-5, 5, (2, 5, 3)).numpy()
    index_tensor = torch.tensor([0, 2]).numpy()
    input_dict = {"input": input_tensor, "dim": 1, "index": index_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: 1D float tensor, dim=0, integer index
    input_tensor = torch.randn(6).numpy()
    index_tensor = torch.tensor([1, 3, 5]).numpy()
    input_dict = {"input": input_tensor, "dim": 0, "index": index_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: 4D complex tensor, dim=2, integer index
    input_tensor = torch.randn(2, 3, 4, 5, dtype=torch.complex64).numpy()
    index_tensor = torch.tensor([0, 2]).numpy()
    input_dict = {"input": input_tensor, "dim": 2, "index": index_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: 2D tensor, dim=1, index containing duplicate values
    input_tensor = torch.randn(4, 4).numpy()
    index_tensor = torch.tensor([1, 1, 3, 0]).numpy()
    input_dict = {"input": input_tensor, "dim": 1, "index": index_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: 5D tensor, dim = 4, mixed positive and negative indices
    input_tensor = torch.randn(2, 2, 2, 2, 2).numpy()
    index_tensor = torch.tensor([0]).numpy()
    input_dict = {"input": input_tensor, "dim": 4, "index": index_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7: Long Tensor Input
    input_tensor = torch.arange(24).reshape(2, 3, 4).long().numpy()
    index_tensor = torch.tensor([0, 2]).numpy()
    input_dict = {"input": input_tensor, "dim": 1, "index": index_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = index_select_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('index_select', generated_inputs)
