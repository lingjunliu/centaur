
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def relu6_inputs():
    list_of_inputs = []

    input_arr = torch.tensor([-1.0, 0.0, 3.5, 6.0, 8.2], dtype=torch.float32).numpy()
    input_dict = {"inplace": False, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.arange(-10, 10, dtype=torch.float32).reshape(4, 5).numpy()
    input_dict = {"inplace": True, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor(-2.5, dtype=torch.float64).numpy()
    input_dict = {"inplace": False, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([[[-3, 0, 6, 7], [9, -1, 2, 6]]], dtype=torch.int32).numpy()
    input_dict = {"inplace": True, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randint(0, 256, (2, 3, 4), dtype=torch.uint8).numpy()
    input_dict = {"inplace": False, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = (torch.randn(2, 3, 4, 5, dtype=torch.float16) * 10).numpy()
    input_dict = {"inplace": True, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.empty(0, dtype=torch.float32).numpy()
    input_dict = {"inplace": False, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.linspace(-12, 12, steps=60, dtype=torch.float32).reshape(1, 2, 3, 2, 5).numpy()
    input_dict = {"inplace": True, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([[-100, -1, 0, 6, 7, 1000]], dtype=torch.int64).numpy()
    input_dict = {"inplace": False, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([1e-6, 1e6, -1e6, 5.9999], dtype=torch.float32).numpy()
    input_dict = {"inplace": True, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.arange(12, dtype=torch.float32).reshape(3, 4).t().numpy()
    input_dict = {"inplace": False, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(3, 3, 3, dtype=torch.float64).numpy()
    input_dict = {"inplace": True, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.ReLU6"] = relu6_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.ReLU6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ReLU6'.")


check_valid('torch.nn.ReLU6', generated_inputs['torch.nn.ReLU6'], lib="torch", suffix=0)
