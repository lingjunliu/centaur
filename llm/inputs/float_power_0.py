
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def float_power_inputs():
    list_of_inputs = []

    input1 = torch.randn(2, 3).numpy()
    exponent1 = 2.0
    input_dict1 = {
        "input": input1,
        "exponent": exponent1,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randint(-5, 5, (3, 4), dtype=torch.int32).numpy()
    exponent2 = 0.5
    input_dict2 = {
        "input": input2,
        "exponent": exponent2,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(1, 5, 5).numpy()
    exponent3 = -1.0
    input_dict3 = {
        "input": input3,
        "exponent": exponent3,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(2, 2, 2, 2).numpy()
    exponent4 = 1.5
    input_dict4 = {
        "input": input4,
        "exponent": exponent4,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(4).numpy()
    exponent5 = 0.0
    input_dict5 = {
        "input": input5,
        "exponent": exponent5,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = (torch.rand(2,3) + 1j*torch.rand(2,3)).numpy()
    exponent6 = 2.0
    input_dict6 = {
        "input": input6,
        "exponent": exponent6,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = torch.tensor([-1.0, 0.0, 1.0]).numpy()
    exponent7 = 3.0
    input_dict7 = {
        "input": input7,
        "exponent": exponent7,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs["torch.float_power"] = float_power_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.float_power' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.float_power'.")

check_valid('torch.float_power', generated_inputs['torch.float_power'], lib="torch")
