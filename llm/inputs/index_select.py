
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def index_select_inputs():
    list_of_inputs = []

    # Input 1: Basic case with positive indices
    input_tensor = torch.randn(3, 4, 5).numpy()
    dim = 1
    index_tensor = torch.tensor([0, 2, 1]).numpy()
    input_dict = {"input": input_tensor, "dim": dim, "index": index_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Negative indices
    input_tensor = torch.randn(3, 4).numpy()
    dim = 0
    index_tensor = torch.tensor([-1]).numpy()
    input_dict = {"input": input_tensor, "dim": dim, "index": index_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D input tensor
    input_tensor = torch.arange(10).float().numpy()
    dim = 0
    index_tensor = torch.tensor([2, 5, 7, 1]).numpy()
    input_dict = {"input": input_tensor, "dim": dim, "index": index_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different data type (int)
    input_tensor = torch.randint(0, 10, (2, 3, 4)).numpy()
    dim = 1
    index_tensor = torch.tensor([0, 1]).numpy()
    input_dict = {"input": input_tensor, "dim": dim, "index": index_tensor}
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
