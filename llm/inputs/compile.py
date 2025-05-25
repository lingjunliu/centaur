
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def compile_inputs():
    list_of_inputs = []

    def simple_model(x):
        return torch.sin(x) + torch.cos(x)

    model = simple_model

    input_dict_1 = {
        "model": model,
        "options": {"triton.cudagraphs": True},
        "dynamic": False,
        "backend": 'inductor',
        "fullgraph": True,
        "mode": 'default',
        "config": None,
        "debug": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    def model_2(x):
        return torch.relu(torch.matmul(x, x.T))

    model = model_2

    input_dict_2 = {
        "model": model,
        "options": {},
        "dynamic": True,
        "backend": 'inductor',
        "fullgraph": False,
        "mode": 'max-autotune',
        "config": None,
        "debug": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    def model_3(x):
        return torch.sigmoid(x * 2)

    model = model_3

    input_dict_3 = {
        "model": model,
        "options": {"epilogue_fusion": True, "max_autotune": True},
        "dynamic": None,
        "backend": 'inductor',
        "fullgraph": False,
        "mode": 'default',
        "config": None,
        "debug": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    def model_4(x):
        return torch.log(torch.abs(x) + 1)

    model = model_4

    input_dict_4 = {
        "model": model,
        "options": {"shape_padding": True},
        "dynamic": False,
        "backend": 'inductor',
        "fullgraph": False,
        "mode": 'reduce-overhead',
        "config": None,
        "debug": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    def model_5(x):
        return torch.sqrt(torch.square(x))

    model = model_5

    input_dict_5 = {
        "model": model,
        "options": {"trace.enabled": True},
        "dynamic": True,
        "backend": 'inductor',
        "fullgraph": True,
        "mode": 'max-autotune-no-cudagraphs',
        "config": None,
        "debug": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

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
