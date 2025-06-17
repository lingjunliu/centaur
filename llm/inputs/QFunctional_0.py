
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def QFunctional_inputs():
    list_of_inputs = []

    # Example 1: Simple QFunctional object with addition
    class Add(torch.nn.quantized.QFunctional):
        def __init__(self):
            super().__init__()
        def forward(self, x, y):
            return torch.ops.quantized.add(x, y, 1.0, 0)

    add_op = Add()
    x = torch.quantize_per_tensor(torch.randn(3, 4), 0.5, 10, torch.quint8)
    y = torch.quantize_per_tensor(torch.randn(3, 4), 0.5, 10, torch.quint8)


    input_dict = {
        "forward": lambda x, y: add_op.forward(x, y),
        "inner":{
            "args": (x, y),
            "kwargs": {}
        }
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.quantized.QFunctional"] = QFunctional_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.quantized.QFunctional' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.quantized.QFunctional'.")

check_valid('torch.nn.quantized.QFunctional', generated_inputs['torch.nn.quantized.QFunctional'], lib="torch")
