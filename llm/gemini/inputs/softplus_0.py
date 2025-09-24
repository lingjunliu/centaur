
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def softplus_inputs():
    list_of_inputs = []

    input_val = torch.randn(3, 4).numpy()
    beta_val = 1.0
    threshold_val = 20.0
    input_dict = {"input": input_val, "beta": beta_val, "threshold": threshold_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_val = torch.randn(2, 2, 2).numpy()
    beta_val = 0.5
    threshold_val = 10.0
    input_dict = {"input": input_val, "beta": beta_val, "threshold": threshold_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_val = torch.randn(5).numpy()
    beta_val = 2.0
    threshold_val = 30.0
    input_dict = {"input": input_val, "beta": beta_val, "threshold": threshold_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_val = torch.randn(1, 1, 1, 1).numpy()
    beta_val = 0.1
    threshold_val = 5.0
    input_dict = {"input": input_val, "beta": beta_val, "threshold": threshold_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_val = torch.randint(-5, 5, (2, 3)).float().numpy()
    beta_val = 1.5
    threshold_val = 25.0
    input_dict = {"input": input_val, "beta": beta_val, "threshold": threshold_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_val = torch.randn(2, 3) * 100.0
    input_val = input_val.numpy()
    beta_val = 1.0
    threshold_val = 20.0
    input_dict = {"input": input_val, "beta": beta_val, "threshold": threshold_val}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_val = torch.randn(10).numpy()
    beta_val = 0.75
    threshold_val = 15.0
    input_dict = {"input": input_val, "beta": beta_val, "threshold": threshold_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.functional.softplus"] = softplus_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.softplus' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.softplus'.")

check_valid('torch.nn.functional.softplus', generated_inputs['torch.nn.functional.softplus'], lib="torch")
