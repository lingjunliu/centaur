
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy
import numpy as np

def cond_inputs():
    list_of_inputs = []

    def true_fn1(x):
        return x + 1

    def false_fn1(x):
        return x - 1

    x1 = torch.tensor([2.0]).numpy()
    input_dict1 = {
        "pred": True,
        "true_fn": true_fn1,
        "false_fn": false_fn1,
        "operands": (x1,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    def true_fn2(x):
        return x * 2

    def false_fn2(x):
        return x / 2

    x2 = torch.tensor([4.0]).numpy()
    input_dict2 = {
        "pred": False,
        "true_fn": true_fn2,
        "false_fn": false_fn2,
        "operands": (x2,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    def true_fn3(x, y):
        return x + y

    def false_fn3(x, y):
        return x - y

    x3 = torch.tensor([5.0]).numpy()
    y3 = torch.tensor([2.0]).numpy()
    input_dict3 = {
        "pred": torch.tensor([True]).bool().numpy().item(),
        "true_fn": true_fn3,
        "false_fn": false_fn3,
        "operands": (x3, y3)
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    def true_fn4():
        return torch.tensor([10.0]).numpy()

    def false_fn4():
        return torch.tensor([5.0]).numpy()
    
    input_dict4 = {
        "pred": torch.tensor([False]).bool().numpy().item(),
        "true_fn": true_fn4,
        "false_fn": false_fn4,
        "operands": ()
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    def true_fn5(x):
        return torch.sin(x)

    def false_fn5(x):
        return torch.cos(x)

    x5 = torch.tensor([np.pi / 2]).numpy()
    input_dict5 = {
        "pred": x5 < 5,
        "true_fn": true_fn5,
        "false_fn": false_fn5,
        "operands": (x5,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    def true_fn6(x):
        return x * 3

    def false_fn6(x):
        return x / 3
    
    x6 = torch.tensor([-9.0]).numpy()

    input_dict6 = {
        "pred": torch.tensor([True]).bool().numpy(),
        "true_fn": true_fn6,
        "false_fn": false_fn6,
        "operands": (x6,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    return list_of_inputs

generated_inputs = cond_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('cond', generated_inputs)
