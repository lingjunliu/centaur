
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import torch.nn as nn
import copy
import numpy as np

def dataparallel_inputs():
    list_of_inputs = []

    class DummyModule(nn.Module):
        def __init__(self):
            super(DummyModule, self).__init__()
            self.linear = nn.Linear(10, 5)

        def forward(self, x):
            return self.linear(x)

    module = DummyModule()

    input_dict = {
        "module": module,
        "device_ids": [],
        "output_device": None,
        "dim": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = dataparallel_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('DataParallel', generated_inputs)
