
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def ZeroPad2d_inputs():
    list_of_inputs = []

    input = torch.randn(1, 1, 3, 3).numpy()
    padding = 2
    input_dict = {"padding": padding, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 1, 3, 3).numpy()
    padding = (1, 1, 2, 0)
    input_dict = {"padding": padding, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(2, 3, 4, 5).numpy()
    padding = (0, 1, 2, 3)
    input_dict = {"padding": padding, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(3, 2, 6, 7).numpy()
    padding = 1
    input_dict = {"padding": padding, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 5, 5).numpy()
    padding = (2, 1, 0, 1)
    input_dict = {"padding": padding, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.randn(1, 1, 2, 2).numpy()
    padding = (0, 0, 0, 0)
    input_dict = {"padding": padding, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.ZeroPad2d_2"] = ZeroPad2d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.ZeroPad2d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ZeroPad2d_2'.")

check_valid('torch.nn.ZeroPad2d', generated_inputs['torch.nn.ZeroPad2d_2'], lib="torch")
