
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def allclose_inputs():
    list_of_inputs = []

    input1 = torch.randn(3, 4).double().numpy()
    input2 = input1 + np.random.normal(0, 1e-7, input1.shape)
    input_dict = {
        "input": input1,
        "other": input2,
        "atol": 1e-08,
        "rtol": 1e-05,
        "equal_nan": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = torch.randn(2, 2).double().numpy()
    input2 = input1 + np.random.normal(0, 1e-6, input1.shape)
    input_dict = {
        "input": input1,
        "other": input2,
        "atol": 1e-07,
        "rtol": 1e-05,
        "equal_nan": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = torch.tensor([float('nan'), 1.0, 2.0], dtype=torch.float64).numpy()
    input2 = torch.tensor([float('nan'), 1.0, 2.0], dtype=torch.float64).numpy()
    input_dict = {
        "input": input1,
        "other": input2,
        "atol": 1e-08,
        "rtol": 1e-05,
        "equal_nan": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = torch.randn(5, 5).double().numpy()
    input2 = input1 + 1
    input_dict = {
        "input": input1,
        "other": input2,
        "atol": 1.1,
        "rtol": 1e-05,
        "equal_nan": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input1 = torch.randn(2, 3, 4).double().numpy()
    input2 = input1 * (1 + 1e-4)
    input_dict = {
        "input": input1,
        "other": input2,
        "atol": 1e-08,
        "rtol": 1e-03,
        "equal_nan": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

list_of_inputs = allclose_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('allclose', list_of_inputs)
