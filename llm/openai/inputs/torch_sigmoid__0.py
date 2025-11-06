
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def sigmoid_inplace_inputs():
    list_of_inputs = []

    input = torch.tensor([-5.0, -1.0, 0.0, 1.0, 5.0], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.randn(2, 3, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.randn(2, 3, 4, dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor(0.0, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    base = torch.arange(12, dtype=torch.float32).view(3, 4).t()
    input = base.numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([-100.0, -50.0, 50.0, 100.0], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([1+1j, -1-2j, 3+0j], dtype=torch.complex64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    r = torch.randn(2, 3, dtype=torch.float64)
    i = torch.randn(2, 3, dtype=torch.float64)
    input = torch.complex(r, i).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.linspace(-10, 10, steps=120, dtype=torch.float32).view(2, 3, 4, 5).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([float('nan'), float('inf'), float('-inf'), 0.0], dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    base = torch.arange(20, dtype=torch.float32).view(4, 5).numpy()
    input = base[:, ::2]
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.empty((0, 3), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.randn(1, 2, 1, 3, 4, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.ones((1, 1, 1), dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    return list_of_inputs

generated_inputs["torch.sigmoid_"] = sigmoid_inplace_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.sigmoid_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sigmoid_'.")


check_valid('torch.sigmoid_', generated_inputs['torch.sigmoid_'], lib="torch", suffix=0)
