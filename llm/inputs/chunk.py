
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def chunk_inputs():
    list_of_inputs = []

    input = torch.randn(4, 4).numpy()
    chunks = 2
    dim = 0
    input_dict = {
        "input": input,
        "chunks": chunks,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(4, 4).numpy()
    chunks = 2
    dim = 1
    input_dict = {
        "input": input,
        "chunks": chunks,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(3, 5, 7).numpy()
    chunks = 3
    dim = 0
    input_dict = {
        "input": input,
        "chunks": chunks,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(3, 5, 7).numpy()
    chunks = 5
    dim = 1
    input_dict = {
        "input": input,
        "chunks": chunks,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(2, 6, 4, 8).numpy()
    chunks = 2
    dim = 3
    input_dict = {
        "input": input,
        "chunks": chunks,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

list_of_inputs = chunk_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('chunk', list_of_inputs)
