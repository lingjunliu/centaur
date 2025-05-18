
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def eq_inputs():
    list_of_inputs = []

    input1 = torch.randn(2, 3).numpy()
    input2 = torch.randn(2, 3).numpy()

    input_dict = {
        "input": input1,
        "other": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = torch.tensor([1, 2, 3]).numpy()
    input2 = torch.tensor([1, 2, 4]).numpy()

    input_dict = {
        "input": input1,
        "other": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = np.array([1.0, 2.0, 3.0])
    input2 = np.array([1.0, 2.0, 3.0])

    input_dict = {
        "input": input1,
        "other": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = torch.randint(0, 10, (5, 5)).numpy()
    input2 = torch.randint(0, 10, (5, 5)).numpy()
    input_dict = {
        "input": input1,
        "other": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = torch.randn(1, 10, 10).numpy()
    input2 = torch.randn(1, 10, 10).numpy()
    input_dict = {
        "input": input1,
        "other": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

list_of_inputs = eq_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('eq', list_of_inputs)
