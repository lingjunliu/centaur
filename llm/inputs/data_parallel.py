
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import torch.nn as nn
import torch.nn.parallel
import copy
import numpy as np

def data_parallel_inputs():
    list_of_inputs = []

    class DummyModule(nn.Module):
        def __init__(self):
            super(DummyModule, self).__init__()
            self.linear = nn.Linear(10, 5)

        def forward(self, x):
            return self.linear(x)

    module = DummyModule()

    input1 = torch.randn(20, 10).float().numpy()
    input_dict1 = {
        "module": module,
        "inputs": input1,
        "device_ids": [0],
        "output_device": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(10, 10).double().numpy()
    input_dict2 = {
        "module": module,
        "inputs": input2,
        "device_ids": [0, 1],
        "output_device": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(5, 10).half().numpy()
    input_dict3 = {
        "module": module,
        "inputs": input3,
        "device_ids": [0],
        "output_device": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(1, 10, dtype=torch.complex64).numpy()
    input_dict4 = {
        "module": module,
        "inputs": input4,
        "device_ids": None,
        "output_device": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(30, 10, dtype=torch.complex128).numpy()
    input_dict5 = {
        "module": module,
        "inputs": input5,
        "device_ids": None,
        "output_device": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.randn(10, 10, 10).float().numpy()
    input_dict6 = {
        "module": module,
        "inputs": input6,
        "device_ids": [0],
        "output_device": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    return list_of_inputs

generated_inputs = data_parallel_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('data_parallel', generated_inputs)
