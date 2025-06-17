
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_compile_inputs():
    list_of_inputs = []

    def simple_func(x):
        return torch.sin(x) + torch.cos(x)

    input_dict = {
        "model": simple_func,
        "fullgraph": False,
        "dynamic": None,
        "backend": "inductor",
        "mode": "default",
        "options": None,
        "disable": False
    }

    def complex_model(x, y):
        z = torch.matmul(x, y)
        z = torch.relu(z)
        z = torch.add(z, 1)
        return z
    
    return list_of_inputs

generated_inputs["torch.compile_1"] = torch_compile_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.compile_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.compile_1'.")

check_valid('torch.compile', generated_inputs['torch.compile_1'], lib="torch")
