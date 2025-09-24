
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def narrow_inputs():
    list_of_inputs = []

    input1 = torch.randn(3, 4, 5).numpy()
    dim1 = 1
    start1 = 0
    length1 = 2
    input_dict1 = {
        "input": input1,
        "dim": dim1,
        "start": start1,
        "length": length1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randint(0, 10, (2, 5, 7)).numpy()
    dim2 = 0
    start2 = 1
    length2 = 1
    input_dict2 = {
        "input": input2,
        "dim": dim2,
        "start": start2,
        "length": length2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(4, 6).numpy()
    dim3 = 1
    start3 = 2
    length3 = 3
    input_dict3 = {
        "input": input3,
        "dim": dim3,
        "start": start3,
        "length": length3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(2, 3, 4, 5).numpy()
    dim4 = 2
    start4 = 1
    length4 = 2
    input_dict4 = {
        "input": input4,
        "dim": dim4,
        "start": start4,
        "length": length4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = torch.randn(5).numpy()
    dim5 = 0
    start5 = 2
    length5 = 2
    input_dict5 = {
        "input": input5,
        "dim": dim5,
        "start": start5,
        "length": length5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.randn(1, 2, 3, 4, 5).numpy()
    dim6 = 3
    start6 = 0
    length6 = 3
    input_dict6 = {
        "input": input6,
        "dim": dim6,
        "start": start6,
        "length": length6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = torch.randn(2,2).numpy()
    dim7 = 0
    start7 = 0
    length7 = 2
    input_dict7 = {
        "input": input7,
        "dim": dim7,
        "start": start7,
        "length": length7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = narrow_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('narrow', generated_inputs)
