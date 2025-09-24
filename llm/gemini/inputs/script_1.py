
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_jit_script_1_inputs():
    list_of_inputs = []

    def func1(x, y):
        if x.max() > y.max():
            r = x
        else:
            r = y
        return r

    input_dict = {
        "obj": func1,
        "optimize": True,
        "example_inputs": [(torch.ones(2, 2), torch.ones(2, 2))]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    def func2(a, b):
        return a + b

    input_dict = {
        "obj": func2,
        "optimize": False,
        "example_inputs": [(torch.tensor(3), torch.tensor(4))]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    class MyModule(torch.nn.Module):
        def __init__(self, N, M):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.rand(N, M))
            self.linear = torch.nn.Linear(N, M)
        def forward(self, input):
            output = self.weight.mv(input)
            output = self.linear(output)
            return output

    input_dict = {
        "obj": MyModule(2,3),
        "optimize": True,
        "example_inputs": [torch.randn(2)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.jit.script_1"] = torch_jit_script_1_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.jit.script_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.jit.script_1'.")

check_valid('torch.jit.script', generated_inputs['torch.jit.script_1'], lib="torch")
