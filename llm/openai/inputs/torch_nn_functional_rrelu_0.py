
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def rrelu_inputs():
    list_of_inputs = []

    # 1
    input_arr = torch.tensor([-1.0, 0.0, 1.0, 2.5], dtype=torch.float32).numpy()
    input_dict = {
        "input": input_arr,
        "lower": 1.0 / 8.0,
        "upper": 1.0 / 3.0,
        "training": True,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 2
    input_arr = torch.randn(2, 3, dtype=torch.float64).numpy()
    input_dict = {
        "input": input_arr,
        "lower": 0.0,
        "upper": 1.0,
        "training": True,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 3
    input_arr = torch.linspace(-2, 2, steps=24, dtype=torch.float16).view(2, 3, 4).numpy()
    input_dict = {
        "input": input_arr,
        "lower": 0.01,
        "upper": 0.1,
        "training": False,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 4
    input_arr = torch.arange(-6, 6, dtype=torch.float32).view(1, 3, 2, 2).numpy()
    input_dict = {
        "input": input_arr,
        "lower": 0.2,
        "upper": 0.2,
        "training": True,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 5
    input_arr = torch.tensor(3.14, dtype=torch.float32).numpy()
    input_dict = {
        "input": input_arr,
        "lower": 0.5,
        "upper": 0.9,
        "training": False,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 6
    input_arr = torch.randn(2, 1, 2, 1, 3, dtype=torch.float32).numpy()
    input_dict = {
        "input": input_arr,
        "lower": 0.001,
        "upper": 0.002,
        "training": False,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 7
    input_arr = torch.arange(-6, 6, dtype=torch.float32).view(3, 4).t().numpy()
    input_dict = {
        "input": input_arr,
        "lower": 0.05,
        "upper": 0.95,
        "training": True,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 8
    input_arr = torch.tensor([-1000.0, -0.001, 0.0, 0.001, 1000.0], dtype=torch.float64).numpy()
    input_dict = {
        "input": input_arr,
        "lower": 0.01,
        "upper": 0.99,
        "training": True,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 9
    input_arr = torch.empty(0, 5, dtype=torch.float32).numpy()
    input_dict = {
        "input": input_arr,
        "lower": 0.1,
        "upper": 0.2,
        "training": False,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 10
    input_arr = torch.tensor([-1.5, -0.5, 0.5, 1.5], dtype=torch.float16).numpy()
    input_dict = {
        "input": input_arr,
        "lower": 0.05,
        "upper": 0.2,
        "training": False,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 11
    input_arr = torch.ones(1, 1, 1, dtype=torch.float64).mul_(-2.0).numpy()
    input_dict = {
        "input": input_arr,
        "lower": 0.33333334,
        "upper": 0.33333335,
        "training": True,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 12
    input_arr = torch.randn(4, 5, 6, dtype=torch.float32).numpy()
    input_dict = {
        "input": input_arr,
        "lower": 0.0,
        "upper": 0.5,
        "training": True,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.functional.rrelu"] = rrelu_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.rrelu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.rrelu'.")


check_valid('torch.nn.functional.rrelu', generated_inputs['torch.nn.functional.rrelu'], lib="torch", suffix=0)
