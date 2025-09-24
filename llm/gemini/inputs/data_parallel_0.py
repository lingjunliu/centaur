
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import torch.nn as nn
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

    # Input 1: Basic case with a float tensor, put module on GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        module = DummyModule().to(device)
        inputs = torch.randn(20, 10).numpy()
        device_ids = [0]
        output_device = 0

        dim = 0
        input_dict = {
            "module": module,
            "inputs": inputs,
            "device_ids": device_ids,
            "output_device": output_device,
            "dim": dim
        }
        list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Empty device_ids, module on CPU
    module = DummyModule()  # Create new module on CPU
    inputs = torch.randn(20, 10).numpy()
    device_ids = []
    output_device = -1
    dim = 0
    input_dict = {
        "module": module,
        "inputs": inputs,
        "device_ids": device_ids,
        "output_device": output_device,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    if not torch.cuda.is_available():
      list_of_inputs.pop(0)


    return list_of_inputs

generated_inputs["torch.nn.parallel.data_parallel"] = data_parallel_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.parallel.data_parallel' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.parallel.data_parallel'.")

check_valid('torch.nn.parallel.data_parallel', generated_inputs['torch.nn.parallel.data_parallel'], lib="torch")
