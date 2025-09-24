
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import torch.nn as nn
import torch.nn.parallel
import copy
import numpy as np

def data_parallel_inputs():
    generated_inputs = []

    class DummyModule(nn.Module):
        def __init__(self):
            super(DummyModule, self).__init__()
            self.linear = nn.Linear(10, 5)

        def forward(self, x):
            return self.linear(x)

    module = DummyModule()

    # Input 1: Simple case with a single GPU
    inputs = torch.randn(20, 10).numpy()
    device_ids = [0]
    output_device = 0
    input_dict = {'module': module, 'inputs': inputs, 'device_ids': device_ids, 'output_device': output_device}
    generated_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Multiple GPUs
    inputs = torch.randn(32, 10).numpy()
    device_ids = [0, 1]
    output_device = 0
    input_dict = {'module': module, 'inputs': inputs, 'device_ids': device_ids, 'output_device': output_device}
    generated_inputs.append(copy.deepcopy(input_dict))

    return generated_inputs

generated_inputs = data_parallel_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('data_parallel', generated_inputs)
