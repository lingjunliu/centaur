
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy
import numpy as np

def compile_inputs():
    list_of_inputs = []

    def simple_model(x):
        return torch.sin(x) + torch.cos(x)

    def model_with_if(x):
        if torch.sum(x) > 0:
            return torch.relu(x)
        else:
            return torch.sigmoid(x)

    def model_with_loop(x):
        y = torch.zeros_like(x)
        for i in range(x.shape[0]):
            y[i] = x[i] * i
        return y
    
    example_input = torch.randn(4, 4)

    input_dict_1 = {
        "model": simple_model,
        "fullgraph": False,
        "dynamic": None,
        "backend": "inductor",
        "mode": "default",
        "options": None,
        "disable": False,
        "args": (example_input,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))
    
    input_dict_2 = {
        "model": model_with_if,
        "fullgraph": False,
        "dynamic": True,
        "backend": "inductor",
        "mode": "reduce-overhead",
        "options": {"triton.cudagraphs": True},
        "disable": False,
        "args": (example_input,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    input_dict_3 = {
        "model": model_with_loop,
        "fullgraph": False,
        "dynamic": False,
        "backend": "inductor",
        "mode": "max-autotune",
        "options": {"epilogue_fusion": True, "max_autotune": True},
        "disable": False,
        "args": (example_input,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    def custom_backend(graph, inps, ops):
        return graph
    
    input_dict_4 = {
        "model": simple_model,
        "fullgraph": False,
        "dynamic": None,
        "backend": custom_backend,
        "mode": "max-autotune-no-cudagraphs",
        "options": {"fallback_random": True},
        "disable": False,
        "args": (example_input,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))
    
    input_dict_5 = {
        "model": model_with_if,
        "fullgraph": False,
        "dynamic": True,
        "backend": "inductor",
        "mode": "default",
        "options": {"shape_padding": True},
        "disable": True,
        "args": (example_input,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.compile_2"] = compile_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.compile_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.compile_2'.")

check_valid('torch.compile', generated_inputs['torch.compile_2'], lib="torch")
