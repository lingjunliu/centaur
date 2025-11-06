
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def logcumsumexp_inputs():
    list_of_inputs = []

    a = torch.tensor([1.0, -2.0, 3.5, 0.0], dtype=torch.float32).numpy()
    input_dict = {"input": a, "dim": 0, "out": np.empty_like(a)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = torch.randn(3, 4, dtype=torch.float32).numpy()
    input_dict = {"input": a, "dim": 1, "out": np.empty_like(a)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = torch.linspace(-5, 5, steps=12, dtype=torch.float64).reshape(3, 4).numpy()
    input_dict = {"input": a, "dim": 0, "out": np.empty_like(a)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = torch.randn(2, 3, 4, dtype=torch.float32).numpy()
    input_dict = {"input": a, "dim": -1, "out": np.zeros_like(a)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = torch.randn(2, 1, 5, dtype=torch.float64).numpy()
    input_dict = {"input": a, "dim": 1, "out": np.empty_like(a)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = torch.randn(2, 3, 4, 5, dtype=torch.float32).numpy()
    input_dict = {"input": a, "dim": -2, "out": np.empty_like(a)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = torch.randn(2, 2, 2, 2, 2, dtype=torch.float32).numpy()
    input_dict = {"input": a, "dim": 3, "out": np.empty_like(a)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = torch.tensor([[1000.0, -1000.0, 0.0],
                      [50.0, 20.0, -50.0]], dtype=torch.float32).numpy()
    input_dict = {"input": a, "dim": 1, "out": np.zeros_like(a)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = torch.arange(30, dtype=torch.float32).view(3, 10).numpy()[:, ::3]
    input_dict = {"input": a, "dim": -1, "out": np.empty_like(a)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = torch.zeros((1, 5), dtype=torch.float32).numpy()
    input_dict = {"input": a, "dim": 0, "out": np.empty_like(a)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = torch.tensor([[[-1.0, -2.0, -3.0],
                       [0.0, 1.0, 2.0]]], dtype=torch.float64).numpy()
    input_dict = {"input": a, "dim": 2, "out": np.empty_like(a)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = torch.randn(2, 1, 3, 1, dtype=torch.float32).numpy()
    input_dict = {"input": a, "dim": 3, "out": np.empty_like(a)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.logcumsumexp"] = logcumsumexp_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.logcumsumexp' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.logcumsumexp'.")


check_valid('torch.logcumsumexp', generated_inputs['torch.logcumsumexp'], lib="torch", suffix=0)
