
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def hardshrink_inputs():
    list_of_inputs = []

    input_arr = torch.tensor([-1.0, -0.4, 0.0, 0.3, 0.6, 1.2], dtype=torch.float32).numpy()
    lambd = 0.5
    input_dict = {"input": input_arr, "lambd": lambd}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.linspace(-2.0, 2.0, steps=6, dtype=torch.float64).reshape(2, 3).numpy()
    lambd = 1.0
    input_dict = {"input": input_arr, "lambd": lambd}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(2, 2, 3, dtype=torch.float16).numpy()
    lambd = 0.2
    input_dict = {"input": input_arr, "lambd": lambd}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(2, 3, 4, 5, dtype=torch.float32).numpy()
    lambd = 0.0
    input_dict = {"input": input_arr, "lambd": lambd}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(1, 2, 2, 2, 2, dtype=torch.float32).numpy()
    lambd = 2.0
    input_dict = {"input": input_arr, "lambd": lambd}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.empty(2, 0, 3, dtype=torch.float32).numpy()
    lambd = 0.75
    input_dict = {"input": input_arr, "lambd": lambd}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor(0.1, dtype=torch.float32).numpy()
    lambd = 0.05
    input_dict = {"input": input_arr, "lambd": lambd}
    list_of_inputs.append(copy.deepcopy(input_dict))

    base = torch.arange(-6, 6, dtype=torch.float32).view(3, 4)
    input_arr = base[:, ::2].numpy()
    lambd = 3.0
    input_dict = {"input": input_arr, "lambd": lambd}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([float("inf"), -float("inf"), float("nan"), 0.3, -0.3, 1e-6, -1e6], dtype=torch.float32).numpy()
    lambd = 0.3
    input_dict = {"input": input_arr, "lambd": lambd}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.linspace(-5.0, 5.0, steps=60, dtype=torch.float32).view(3, 4, 5).numpy()
    lambd = 4.0
    input_dict = {"input": input_arr, "lambd": lambd}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([-0.5, 0.5, 1.5, -1.5, 0.0], dtype=torch.float64).numpy()
    lambd = 0.5
    input_dict = {"input": input_arr, "lambd": lambd}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.zeros(3, 3, dtype=torch.float32).numpy()
    lambd = 0.1
    input_dict = {"input": input_arr, "lambd": lambd}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.functional.hardshrink"] = hardshrink_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.hardshrink' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.hardshrink'.")


check_valid('torch.nn.functional.hardshrink', generated_inputs['torch.nn.functional.hardshrink'], lib="torch", suffix=0)
