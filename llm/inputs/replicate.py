
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import torch.nn as nn
import copy
import numpy as np

def replicate_inputs():
    list_of_inputs = []

    class SimpleModule(nn.Module):
        def __init__(self):
            super(SimpleModule, self).__init__()
            self.linear = nn.Linear(10, 5)

        def forward(self, x):
            return self.linear(x)

    module1 = SimpleModule()
    device_ids1 = [0]
    input_dict1 = {"module": module1, "device_ids": device_ids1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    return list_of_inputs

generated_inputs = replicate_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('replicate', generated_inputs)
