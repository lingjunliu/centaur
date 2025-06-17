
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import torch.nn as nn
import copy
import numpy as np

def DataParallel_inputs():
    list_of_inputs = []

    class DummyModule(nn.Module):
        def __init__(self):
            super(DummyModule, self).__init__()
            self.linear = nn.Linear(10, 10)

        def forward(self, x):
            return self.linear(x)

    module = DummyModule()
    device_ids = [0] if torch.cuda.is_available() else []
    output_device = 0 if torch.cuda.is_available() else None
    dim = 0

    input_dict = {
        "module": module,
        "device_ids": device_ids,
        "output_device": output_device,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.DataParallel"] = DataParallel_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.DataParallel' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.DataParallel'.")

check_valid('torch.nn.DataParallel', generated_inputs['torch.nn.DataParallel'], lib="torch")
