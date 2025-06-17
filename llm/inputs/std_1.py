
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_std_inputs():
    list_of_inputs = []

    # Case 1: Basic case with dim=1, keepdim=True
    a = torch.tensor([[0.2035, 1.2959, 1.8101, -0.4644],
                      [1.5027, -0.3270, 0.5905, 0.6538],
                      [-1.5745, 1.3330, -0.5596, -0.6548],
                      [0.1264, -0.5080, 1.6420, 0.1992]]).numpy()
    input_dict = {
        "input": a,
        "dim": 1,
        "correction": 1,
        "keepdim": True,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: dim=0, keepdim=False
    a = torch.randn(3, 4).numpy()
    input_dict = {
        "input": a,
        "dim": 0,
        "correction": 0,
        "keepdim": False,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: No dim specified, all dimensions reduced
    a = torch.randn(2, 3, 4).numpy()
    input_dict = {
        "input": a,
        "dim": None,
        "correction": 1,
        "keepdim": False,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Multiple dimensions specified
    a = torch.randn(2, 3, 4, 5).numpy()
    input_dict = {
        "input": a,
        "dim": (0, 2),
        "correction": 1,
        "keepdim": True,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Negative values and different correction
    a = torch.randn(5, 5) * -1.0
    a = a.numpy()
    input_dict = {
        "input": a,
        "dim": 1,
        "correction": 2,
        "keepdim": False,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.std_1"] = torch_std_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.std_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.std_1'.")

check_valid('torch.std', generated_inputs['torch.std_1'], lib="torch")
