
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def floor__inputs():
    list_of_inputs = []

    input = torch.tensor([1.2, -3.7, 0.0, 5.999, -0.0001], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[1.5, 2.0, -2.1],
                          [3.7, -4.2, 0.9]], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor(-123.456, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.randn(2, 3, 4, dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.empty((0,), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([-10, 0, 1, 2, 255], dtype=torch.int64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[0, 255],
                          [128, 64]], dtype=torch.uint8).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.arange(-6, 6, dtype=torch.int32).reshape(2, 2, 3).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([float('nan'), float('inf'), float('-inf'), -1.001, 2.999], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.linspace(-5, 5, steps=24, dtype=torch.float32).reshape(2, 1, 3, 4).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.empty((0, 3), dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([1e-7, -1e-7, 3.0000001, -3.0000001], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[1.9, -2.9],
                          [3.2, -4.8]], dtype=torch.float32).T.numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    return list_of_inputs

generated_inputs["torch.floor_"] = floor__inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.floor_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.floor_'.")


check_valid('torch.floor_', generated_inputs['torch.floor_'], lib="torch", suffix=0)
