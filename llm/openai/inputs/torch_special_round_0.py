
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def special_round_inputs():
    list_of_inputs = []

    input = torch.tensor([1.2, 2.5, -3.7], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[0.5, 1.5, 2.5],
                          [3.5, -0.5, -1.5]], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[[0.1, 0.9, 1.1],
                           [1.5, 2.5, -2.5]],
                          [[-0.1, -0.9, -1.1],
                           [1000.4, 1000.5, -1000.5]]], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor(2.5, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.empty((0,), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([1e10 + 0.5, -(1e10 + 0.5), 1e12 + 0.49, -(1e12 + 0.49)], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([float('inf'), float('-inf'), float('nan')], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    t = torch.arange(-12, 12, dtype=torch.float32).reshape(2, 1, 3, 4) / 3.0
    input = t.numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    base = torch.arange(12, dtype=torch.float32).reshape(3, 4) + 0.5
    input = base.numpy().T
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([2**60 + 0.5, -(2**60 + 0.5), 2**53 + 0.5], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([0.0, -0.0, 1e-7, -1e-7, 1.4999, 1.5, -1.5], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    return list_of_inputs

generated_inputs["torch.special.round"] = special_round_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.special.round' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.round'.")


check_valid('torch.special.round', generated_inputs['torch.special.round'], lib="torch", suffix=0)
