
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import torch.nn as nn
from typing import List, Tuple, Dict, Callable

def torch_jit_script_inputs():
    list_of_inputs = []

    # 1. Scripting a function with decorator
    def foo(x, y):
        if x.max() > y.max():
            r = x
        else:
            r = y
        return r

    input_dict = {
        "obj": foo,
        "optimize": True,
        "example_inputs": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 2. Scripting a function with example_inputs
    def test_sum(a, b):
        return a + b

    input_dict = {
        "obj": test_sum,
        "optimize": False,
        "example_inputs": [(torch.tensor(3), torch.tensor(4))]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 3. Scripting an nn.Module
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
        "obj": MyModule(2, 3),
        "optimize": True,
        "example_inputs": [(torch.randn(3),)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    class TestNNModule(torch.nn.Module):
        def forward(self, a: List[int]) -> List[int]:
            result = a
            return result
    
    pdt_model = TestNNModule()
    
    input_dict = {
        "obj": pdt_model,
        "optimize": False,
        "example_inputs": [(torch.tensor([10, 20]),)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.jit.script_2"] = torch_jit_script_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.jit.script_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.jit.script_2'.")

check_valid('torch.jit.script', generated_inputs['torch.jit.script_2'], lib="torch")
