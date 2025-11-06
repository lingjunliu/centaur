
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, numpy as np, copy

def trunc__inputs():
    list_of_inputs = []

    input = torch.tensor([1.9, -2.1, 0.0, 3.999], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[np.inf, -np.inf, np.nan],
                          [5.5, -5.5, 0.1]], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor(-3.75, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[[1.5, -1.5], [2.9, -2.9]],
                          [[0.49, -0.49], [1000.9, -1000.9]]], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.arange(-5, 5, dtype=torch.int32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[1234567890123, -1234567890123], [0, -1]], dtype=torch.int64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    base = torch.linspace(-5, 5, steps=11, dtype=torch.float32).numpy()
    input = base[::2]
    list_of_inputs.append(copy.deepcopy({"input": input}))

    base2 = torch.arange(12, dtype=torch.float32).reshape(3, 4).numpy()
    input = base2.T
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.empty((0, 3), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([1e20, -1e20, 1e-4, -1e-4], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = (torch.arange(24, dtype=torch.float32).reshape(2, 3, 2, 2) / 3.0 - 2.0).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[0.9999, -0.9999, 10.0001],
                          [-10.0001, 100.5, -100.5]], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    return list_of_inputs

generated_inputs["torch.trunc_"] = trunc__inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.trunc_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.trunc_'.")


check_valid('torch.trunc_', generated_inputs['torch.trunc_'], lib="torch", suffix=0)
