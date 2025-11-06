
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def pdist_inputs():
    list_of_inputs = []

    input = torch.tensor([[0.0, 1.0],
                          [2.0, 3.0]], dtype=torch.float32).numpy()
    p = 2.0
    list_of_inputs.append(copy.deepcopy({"input": input, "p": p}))

    input = torch.tensor([[1.5, -2.0, 3.0],
                          [-4.5, 5.5, -6.5],
                          [7.0, -8.0, 9.0]], dtype=torch.float64).numpy()
    p = 1.0
    list_of_inputs.append(copy.deepcopy({"input": input, "p": p}))

    input = torch.randn(4, 5, dtype=torch.float32).numpy()
    p = 0.0
    list_of_inputs.append(copy.deepcopy({"input": input, "p": p}))

    input = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32).unsqueeze(0).numpy()
    p = 2.0
    list_of_inputs.append(copy.deepcopy({"input": input, "p": p}))

    input = torch.linspace(-2.0, 2.0, steps=5, dtype=torch.float32).reshape(5, 1).numpy()
    p = float("inf")
    list_of_inputs.append(copy.deepcopy({"input": input, "p": p}))

    base = torch.arange(24.0, dtype=torch.float32).reshape(6, 4)
    input = base[::2].numpy()
    p = 3.5
    list_of_inputs.append(copy.deepcopy({"input": input, "p": p}))

    input = torch.tensor([[0.0, 0.0, 0.0],
                          [1e6, -1e6, 1e6],
                          [-1e3, 2e3, -3e3]], dtype=torch.float32).numpy()
    p = 1.5
    list_of_inputs.append(copy.deepcopy({"input": input, "p": p}))

    input = torch.tensor([[float("nan"), 1.0],
                          [float("inf"), -float("inf")],
                          [0.0, -0.0]], dtype=torch.float32).numpy()
    p = 2.0
    list_of_inputs.append(copy.deepcopy({"input": input, "p": p}))

    input = torch.arange(100.0, dtype=torch.float64).reshape(10, 10).numpy()
    p = 7.0
    list_of_inputs.append(copy.deepcopy({"input": input, "p": p}))

    input = torch.tensor([[1.0, -1.0, 2.0],
                          [1.0, -1.0, 2.0],
                          [-2.0, 2.0, -4.0],
                          [3.0, 0.0, -3.0]], dtype=torch.float32).numpy()
    p = 2.0
    list_of_inputs.append(copy.deepcopy({"input": input, "p": p}))

    return list_of_inputs

generated_inputs["torch.pdist"] = pdist_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.pdist' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.pdist'.")


check_valid('torch.pdist', generated_inputs['torch.pdist'], lib="torch", suffix=0)
