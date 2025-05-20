
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def compile_inputs():
    list_of_inputs = []

    def foo(x):
        return torch.sin(x) + torch.cos(x)

    # Input 1: Basic example with fullgraph=True
    input_dict = {
        "model": foo,
        "fullgraph": True,
        "dynamic": None,
        "backend": "inductor",
        "mode": "default",
        "options": {"triton.cudagraphs": True},
        "config": None,
        "debug": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2:  fullgraph=False, dynamic=True
    def bar(x, y):
        return torch.matmul(x, y)

    input_dict = {
        "model": bar,
        "fullgraph": False,
        "dynamic": True,
        "backend": "inductor",
        "mode": "max-autotune",
        "options": {},
        "config": None,
        "debug": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3:  Different backend and mode
    def baz(x):
        return torch.relu(x)

    input_dict = {
        "model": baz,
        "fullgraph": False,
        "dynamic": False,
        "backend": "inductor",
        "mode": "reduce-overhead",
        "options": {},
        "config": None,
        "debug": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: with options
    def qux(x):
        return x * x

    input_dict = {
        "model": qux,
        "fullgraph": False,
        "dynamic": None,
        "backend": "inductor",
        "mode": "max-autotune",
        "options": {"epilogue_fusion": True, "max_autotune": True},
        "config": None,
        "debug": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5:  lambda function as model
    def fun_lambda(x):
        return x + 1

    input_dict = {
        "model": fun_lambda,
        "fullgraph": True,
        "dynamic": False,
        "backend": "inductor",
        "mode": "default",
        "options": {},
        "config": None,
        "debug": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = compile_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('compile', generated_inputs)
