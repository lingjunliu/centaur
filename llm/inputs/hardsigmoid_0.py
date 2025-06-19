
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def hardsigmoid_inputs():
    list_of_inputs = []

    input_1 = torch.randn(3, 4).numpy()
    input_dict_1 = {
        "input": input_1,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    input_2 = torch.randn(2, 2, 2).numpy()
    input_dict_2 = {
        "input": input_2,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    input_3 = torch.randint(-5, 5, (5,)).float().numpy()
    input_dict_3 = {
        "input": input_3,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    input_4 = torch.randn(1, 5, 5, 5).numpy()
    input_dict_4 = {
        "input": input_4,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    input_5 = torch.tensor([-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]).numpy()
    input_dict_5 = {
        "input": input_5,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))
    
    input_6 = torch.zeros(2, 3).numpy()
    input_dict_6 = {
        "input": input_6,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    input_7 = torch.ones(4, 2).numpy()
    input_dict_7 = {
        "input": input_7,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    return list_of_inputs

generated_inputs["torch.nn.functional.hardsigmoid"] = hardsigmoid_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.hardsigmoid' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.hardsigmoid'.")

check_valid('torch.nn.functional.hardsigmoid', generated_inputs['torch.nn.functional.hardsigmoid'], lib="torch")
