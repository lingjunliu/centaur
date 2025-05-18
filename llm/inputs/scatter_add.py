
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def scatter_add_inputs():
    list_of_inputs = []

    input = torch.zeros(5, 3).numpy()
    dim = 0
    index = torch.tensor([[0, 1, 2], [0, 2, 4], [1, 4, 3], [1, 2, 3], [2, 3, 4]]).numpy()
    src = torch.randn(5, 3).numpy()

    input_dict = {
        "input": input,
        "dim": dim,
        "index": index,
        "src": src
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.zeros(3, 5).numpy()
    dim = 1
    index = torch.tensor([[0, 1, 2, 0, 0], [0, 2, 4, 1, 1], [1, 4, 3, 2, 2]]).numpy()
    src = torch.randn(3, 5).numpy()

    input_dict = {
        "input": input,
        "dim": dim,
        "index": index,
        "src": src
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.zeros(2, 3, 4).numpy()
    dim = 0
    index = torch.tensor([[[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]], [[1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1]]]).numpy()
    src = torch.randn(2, 3, 4).numpy()

    input_dict = {
        "input": input,
        "dim": dim,
        "index": index,
        "src": src
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.zeros(2, 3, 4).numpy()
    dim = 1
    index = torch.tensor([[[0, 1, 0, 1], [1, 0, 1, 0]], [[0, 0, 1, 1], [1, 1, 0, 0]]]).numpy()
    src = torch.randn(2, 2, 4).numpy()

    input_dict = {
        "input": input,
        "dim": dim,
        "index": index,
        "src": src
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.zeros(2, 3, 4).numpy()
    dim = 2
    index = torch.tensor([[[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]], [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]]).numpy()
    src = torch.randn(2, 3, 4).numpy()

    input_dict = {
        "input": input,
        "dim": dim,
        "index": index,
        "src": src
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

list_of_inputs = scatter_add_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('scatter_add', list_of_inputs)
