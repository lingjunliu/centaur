
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def torch_special_entr_inputs():
    list_of_inputs = []

    input = torch.tensor([0.1, 0.5, 1.0, 2.0], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([-1.0, -0.5, 0.0, 0.5, 2.0], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[1e-6, 1e-3],
                          [2e-1, 8e-1]], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[[0.0, 0.1],
                           [0.5, 1.0]],
                          [[2.0, 4.0],
                           [8.0, 16.0]]], dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor(0.0, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor(-3.5, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([float('inf'), float('-inf'), float('nan'), 1.0, 0.0, -1.0], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([1e10, 1e20, 1e100, 1e308], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    base = (torch.arange(12, dtype=torch.float32).reshape(3, 4) - 5.0).t()
    input = base.numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.linspace(-2, 2, steps=9, dtype=torch.float32)[::2].numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.empty((2, 0), dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([1e-300, 1e-200, 1e-100, 1e-50, 1e-20], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[[[0.0], [0.2], [1.0]]],
                          [[[2.0], [3.0], [10.0]]]], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    return list_of_inputs

generated_inputs["torch.special.entr"] = torch_special_entr_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.special.entr' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.entr'.")


check_valid('torch.special.entr', generated_inputs['torch.special.entr'], lib="torch", suffix=0)
