
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def rrelu_inputs():
    list_of_inputs = []

    input1 = torch.randn(2).numpy()
    input_dict1 = {
        "lower": 0.1,
        "upper": 0.3,
        "inplace": False,
        "input": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(3, 4).numpy()
    input_dict2 = {
        "lower": 0.2,
        "upper": 0.4,
        "inplace": True,
        "input": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(5, 2, 3).numpy()
    input_dict3 = {
        "lower": -0.1,
        "upper": 0.1,
        "inplace": False,
        "input": input3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(1).numpy()
    input_dict4 = {
        "lower": 0.0,
        "upper": 0.5,
        "inplace": True,
        "input": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(4, 4, 4, 4).numpy()
    input_dict5 = {
        "lower": 0.15,
        "upper": 0.35,
        "inplace": False,
        "input": input5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.randn(2, 2).numpy()
    input_dict6 = {
        "lower": -0.2,
        "upper": -0.1,
        "inplace": True,
        "input": input6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = torch.randn(10).numpy()
    input_dict7 = {
        "lower": 0.125,
        "upper": 0.3333333333333333,
        "inplace": False,
        "input": input7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = torch.randn(3, 3, 3).numpy()
    input_dict8 = {
        "lower": 0.5,
        "upper": 0.6,
        "inplace": True,
        "input": input8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    input9 = torch.randn(2, 5).numpy()
    input_dict9 = {
        "lower": -0.5,
        "upper": 0.0,
        "inplace": False,
        "input": input9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = torch.randn(1, 1, 1, 1).numpy()
    input_dict10 = {
        "lower": 0.05,
        "upper": 0.25,
        "inplace": True,
        "input": input10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.nn.RReLU"] = rrelu_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.RReLU' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.RReLU'.")


check_valid('torch.nn.RReLU', generated_inputs['torch.nn.RReLU'], lib="torch", suffix=0)
