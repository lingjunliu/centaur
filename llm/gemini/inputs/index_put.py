
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def index_put_inputs():
    list_of_inputs = []

    # Input 1: Basic example with float tensor, boolean indices, float values
    input_tensor = torch.randn(5, 5).numpy()
    indices = ([torch.tensor([0, 2, 4]).numpy(), torch.tensor([1, 3, 0]).numpy()],)
    values = torch.randn(3).numpy()
    accumulate = False

    input_dict = {
        "input": input_tensor,
        "indices": indices,
        "values": values,
        "accumulate": accumulate
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Integer tensor, Long indices, Integer values, accumulate=True
    input_tensor = torch.randint(0, 10, (3, 3)).numpy()
    indices = ([torch.tensor([0, 1]).long().numpy(), torch.tensor([1, 2]).long().numpy()],)
    values = torch.randint(0, 5, (2,)).numpy()
    accumulate = True

    input_dict = {
        "input": input_tensor,
        "indices": indices,
        "values": values,
        "accumulate": accumulate
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D tensor, mixed indices, values as tensor, accumulate=False
    input_tensor = torch.randn(2, 3, 4).numpy()
    indices = ([torch.tensor([0, 1]).numpy(), torch.tensor([0, 1]).numpy(), torch.tensor([1, 2]).numpy()],)
    values = torch.randn(2).numpy()
    accumulate = False
    input_dict = {
        "input": input_tensor,
        "indices": indices,
        "values": values,
        "accumulate": accumulate
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D tensor, single index, single value
    input_tensor = torch.arange(5).float().numpy()
    indices = ([torch.tensor([2]).numpy()],)
    values = torch.tensor([10.0]).numpy()
    accumulate = False
    input_dict = {
        "input": input_tensor,
        "indices": indices,
        "values": values,
        "accumulate": accumulate
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative indices, float tensor
    input_tensor = torch.randn(4, 4).numpy()
    indices = ([torch.tensor([0, -1]).numpy(), torch.tensor([1, -2]).numpy()],)
    values = torch.randn(2).numpy()
    accumulate = False
    input_dict = {
        "input": input_tensor,
        "indices": indices,
        "values": values,
        "accumulate": accumulate
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: boolean tensor, boolean indices, boolean values
    input_tensor = torch.randint(0, 2, (2, 2), dtype=torch.bool).numpy()
    indices = ([torch.tensor([0, 1]).numpy(), torch.tensor([0, 1]).numpy()],)
    values = torch.tensor([True, False]).bool().numpy()
    accumulate = False
    input_dict = {
        "input": input_tensor,
        "indices": indices,
        "values": values,
        "accumulate": accumulate
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = index_put_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('index_put', generated_inputs)
