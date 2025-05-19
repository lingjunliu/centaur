
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def chunk_inputs():
    list_of_inputs = []

    input1 = torch.randn(4, 4).numpy()
    chunks1 = 2
    dim1 = 0
    input_dict1 = {
        "input": input1,
        "chunks": chunks1,
        "dim": dim1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(3, 5, 7).numpy()
    chunks2 = 3
    dim2 = 1
    input_dict2 = {
        "input": input2,
        "chunks": chunks2,
        "dim": dim2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randint(0, 10, (2, 6, 4)).numpy()
    chunks3 = 4
    dim3 = 2
    input_dict3 = {
        "input": input3,
        "chunks": chunks3,
        "dim": dim3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(2, 2, 2, 2).numpy()
    chunks4 = 2
    dim4 = 3
    input_dict4 = {
        "input": input4,
        "chunks": chunks4,
        "dim": dim4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(5,).numpy()
    chunks5 = 5
    dim5 = 0
    input_dict5 = {
        "input": input5,
        "chunks": chunks5,
        "dim": dim5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = torch.randn(2, 3, 4, 5).numpy()
    chunks6 = 1
    dim6 = 0
    input_dict6 = {
        "input": input6,
        "chunks": chunks6,
        "dim": dim6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = torch.randn(6, 8).numpy()
    chunks7 = 4
    dim7 = 1
    input_dict7 = {
        "input": input7,
        "chunks": chunks7,
        "dim": dim7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = torch.randn(2, 4, 6, 8).numpy()
    chunks8 = 2
    dim8 = 2
    input_dict8 = {
        "input": input8,
        "chunks": chunks8,
        "dim": dim8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    return list_of_inputs

generated_inputs = chunk_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('chunk', generated_inputs)
