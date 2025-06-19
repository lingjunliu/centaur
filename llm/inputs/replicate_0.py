
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import torch.nn as nn
import copy
import numpy as np

def replicate_inputs():
    list_of_inputs = []

    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_ids = [torch.device('cuda', i) for i in range(min(torch.cuda.device_count(), 2))]
    else:
        device = torch.device('cpu')
        device_ids = [torch.device('cpu')]

    # Input 1: Simple linear layer
    module = nn.Linear(10, 5).to(device)
    input_dict = {"module": module, "device_ids": device_ids}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: More complex sequential model
    module = nn.Sequential(
        nn.Linear(20, 10),
        nn.ReLU(),
        nn.Linear(10, 5)
    ).to(device)
    input_dict = {"module": module, "device_ids": device_ids}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3:  Custom module
    class MyModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(5, 2)

        def forward(self, x):
            return self.linear(x)

    module = MyModule().to(device)
    input_dict = {"module": module, "device_ids": device_ids}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4:  Module with batchnorm
    module = nn.Sequential(
        nn.Linear(10, 20),
        nn.BatchNorm1d(20),
        nn.ReLU(),
        nn.Linear(20, 5)
    ).to(device)
    input_dict = {"module": module, "device_ids": device_ids}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.parallel.replicate"] = replicate_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.parallel.replicate' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.parallel.replicate'.")

check_valid('torch.nn.parallel.replicate', generated_inputs['torch.nn.parallel.replicate'], lib="torch")
