
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def count_nonzero_inputs():
    list_of_inputs = []

    input_tensor = torch.tensor([0, 1, 2, 3, 4, 0, 5, 6])
    dim = None
    input_dict = {"input": input_tensor.numpy(), "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.tensor([[0, 1, 2, 0], [3, 0, 5, 6]])
    dim = None
    input_dict = {"input": input_tensor.numpy(), "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.tensor([[0, 1, 2, 0], [3, 0, 5, 6]])
    dim = (0,)
    input_dict = {"input": input_tensor.numpy(), "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.tensor([[0, 1, 2, 0], [3, 0, 5, 6]])
    dim = (1,)
    input_dict = {"input": input_tensor.numpy(), "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.tensor([[[0, 1, 0], [2, 0, 3]], [[4, 5, 0], [0, 6, 7]]])
    dim = (1,)
    input_dict = {"input": input_tensor.numpy(), "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

list_of_inputs = count_nonzero_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('countNonzero', list_of_inputs)
