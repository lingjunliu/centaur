
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def cond_inputs():
    list_of_inputs = []

    def true_fn1(x):
        return x + 1
    def false_fn1(x):
        return x - 1
    x = torch.tensor([5.0]).numpy()
    input_dict = {
        "pred": True,
        "true_fn": true_fn1,
        "false_fn": false_fn1,
        "operands": (x,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    def true_fn2(x):
        return x * 2
    def false_fn2(x):
        return x / 2
    x = torch.tensor([2.0]).numpy()
    input_dict = {
        "pred": False,
        "true_fn": true_fn2,
        "false_fn": false_fn2,
        "operands": (x,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    def true_fn3(x, y):
      return x + y
    def false_fn3(x, y):
      return x - y
    x = torch.tensor([3.0]).numpy()
    y = torch.tensor([2.0]).numpy()

    input_dict = {
        "pred": torch.tensor([True]).bool().numpy(),
        "true_fn": true_fn3,
        "false_fn": false_fn3,
        "operands": (x, y)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    def true_fn4(x):
        return torch.sin(x)
    def false_fn4(x):
        return torch.cos(x)
    x = torch.tensor([np.pi/2]).numpy()
    input_dict = {
        "pred": torch.tensor([0]).bool().numpy(),
        "true_fn": true_fn4,
        "false_fn": false_fn4,
        "operands": (x,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    def true_fn5(x):
        return x.mean()
    def false_fn5(x):
        return x.sum()
    x = torch.randn(2, 3).numpy()
    input_dict = {
        "pred": x.shape[0] > 1,
        "true_fn": true_fn5,
        "false_fn": false_fn5,
        "operands": (x,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    def true_fn6(x, y):
        return x @ y
    def false_fn6(x, y):
        return x + y

    x = torch.randn(2, 3).numpy()
    y = torch.randn(3, 4).numpy()
    input_dict = {
        "pred": True,
        "true_fn": true_fn6,
        "false_fn": false_fn6,
        "operands": (x, y)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.cond"] = cond_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.cond' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.cond'.")

check_valid('torch.cond', generated_inputs['torch.cond'], lib="torch")
