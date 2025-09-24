
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def ReflectionPad2d_inputs():
    list_of_inputs = []

    input = torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3).numpy()
    padding = 2
    input_dict = {
        "input": input,
        "padding": padding,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 3, 5, 5).numpy()
    padding = 1
    input_dict = {
        "input": input,
        "padding": padding,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(2, 3, 4, 4).numpy()
    padding = (1, 2, 0, 1)
    input_dict = {
        "input": input,
        "padding": padding,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 1, 7, 7).numpy()
    padding = (2, 2, 1, 1)
    input_dict = {
        "input": input,
        "padding": padding,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(3, 5, 6).numpy()
    padding = (1, 0, 2, 1)
    input_dict = {
        "input": input,
        "padding": padding,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 2, 4, 4).numpy()
    padding = (0, 1, 1, 0)
    input_dict = {
        "input": input,
        "padding": padding,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 1, 5, 5).numpy()
    padding = 0
    input_dict = {
        "input": input,
        "padding": padding,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.ReflectionPad2d_1"] = ReflectionPad2d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.ReflectionPad2d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ReflectionPad2d_1'.")

check_valid('torch.nn.ReflectionPad2d', generated_inputs['torch.nn.ReflectionPad2d_1'], lib="torch")
