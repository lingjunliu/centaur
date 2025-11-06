
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def group_norm_inputs():
    list_of_inputs = []

    # Input 1
    x = torch.randn(2, 4, 3, 3, dtype=torch.float32).numpy()
    num_groups = 2
    weight = torch.linspace(0.5, 1.5, steps=4, dtype=torch.float32).numpy()
    bias = torch.zeros(4, dtype=torch.float32).numpy()
    eps = 1e-5
    list_of_inputs.append(copy.deepcopy({
        "input": x, "num_groups": num_groups, "weight": weight, "bias": bias, "eps": eps
    }))

    # Input 2
    x = torch.randn(1, 6, 2, 2, 2, dtype=torch.double).numpy()
    num_groups = 3
    weight = torch.randn(6, dtype=torch.double).numpy()
    bias = torch.randn(6, dtype=torch.double).numpy()
    eps = 1e-5
    list_of_inputs.append(copy.deepcopy({
        "input": x, "num_groups": num_groups, "weight": weight, "bias": bias, "eps": eps
    }))

    # Input 3
    x = torch.randn(3, 8, 5, dtype=torch.float16).numpy()
    num_groups = 4
    weight = torch.ones(8, dtype=torch.float16).numpy()
    bias = torch.zeros(8, dtype=torch.float16).numpy()
    eps = 1e-3
    list_of_inputs.append(copy.deepcopy({
        "input": x, "num_groups": num_groups, "weight": weight, "bias": bias, "eps": eps
    }))

    # Input 4
    x = torch.randn(4, 2, 7, 5, dtype=torch.float32).numpy()
    num_groups = 1
    weight = torch.tensor([-1.0, 0.5], dtype=torch.float32).numpy()
    bias = torch.tensor([0.1, -0.2], dtype=torch.float32).numpy()
    eps = 1e-5
    list_of_inputs.append(copy.deepcopy({
        "input": x, "num_groups": num_groups, "weight": weight, "bias": bias, "eps": eps
    }))

    # Input 5
    x = torch.randn(2, 5, 1, 1, dtype=torch.float32).numpy()
    num_groups = 5
    weight = torch.arange(5, dtype=torch.float32).numpy()
    bias = (-torch.arange(5, dtype=torch.float32)).numpy()
    eps = 1e-6
    list_of_inputs.append(copy.deepcopy({
        "input": x, "num_groups": num_groups, "weight": weight, "bias": bias, "eps": eps
    }))

    # Input 6
    x = torch.randn(1, 12, 2, 3, 4, dtype=torch.float32).numpy()
    num_groups = 6
    weight = torch.randn(12, dtype=torch.float32).numpy()
    bias = torch.zeros(12, dtype=torch.float32).numpy()
    eps = 1e-4
    list_of_inputs.append(copy.deepcopy({
        "input": x, "num_groups": num_groups, "weight": weight, "bias": bias, "eps": eps
    }))

    # Input 7
    x = torch.randn(5, 10, 7, dtype=torch.double).numpy()
    num_groups = 5
    weight = torch.ones(10, dtype=torch.double).numpy()
    bias = (0.01 * torch.arange(10, dtype=torch.double)).numpy()
    eps = 1e-5
    list_of_inputs.append(copy.deepcopy({
        "input": x, "num_groups": num_groups, "weight": weight, "bias": bias, "eps": eps
    }))

    # Input 8 (zero batch)
    x = torch.randn(0, 4, 3, 3, dtype=torch.float32).numpy()
    num_groups = 2
    weight = torch.ones(4, dtype=torch.float32).numpy()
    bias = torch.zeros(4, dtype=torch.float32).numpy()
    eps = 1e-5
    list_of_inputs.append(copy.deepcopy({
        "input": x, "num_groups": num_groups, "weight": weight, "bias": bias, "eps": eps
    }))

    # Input 9 (single channel)
    x = torch.randn(2, 1, 4, 4, dtype=torch.float32).numpy()
    num_groups = 1
    weight = torch.tensor([2.0], dtype=torch.float32).numpy()
    bias = torch.tensor([-1.0], dtype=torch.float32).numpy()
    eps = 1e-8
    list_of_inputs.append(copy.deepcopy({
        "input": x, "num_groups": num_groups, "weight": weight, "bias": bias, "eps": eps
    }))

    # Input 10 (larger eps)
    x = torch.randn(3, 9, 2, 2, dtype=torch.float32).numpy()
    num_groups = 3
    weight = torch.randn(9, dtype=torch.float32).numpy()
    bias = torch.randn(9, dtype=torch.float32).numpy()
    eps = 0.5
    list_of_inputs.append(copy.deepcopy({
        "input": x, "num_groups": num_groups, "weight": weight, "bias": bias, "eps": eps
    }))

    # Input 11 (6D input)
    x = torch.randn(1, 16, 2, 2, 2, 2, dtype=torch.float32).numpy()
    num_groups = 8
    weight = torch.linspace(-1.0, 1.0, steps=16, dtype=torch.float32).numpy()
    bias = torch.zeros(16, dtype=torch.float32).numpy()
    eps = 1e-5
    list_of_inputs.append(copy.deepcopy({
        "input": x, "num_groups": num_groups, "weight": weight, "bias": bias, "eps": eps
    }))

    # Input 12 (more channels)
    x = torch.randn(2, 24, 3, 3, dtype=torch.float32).numpy()
    num_groups = 12
    weight = torch.sin(torch.linspace(0, 3.14159, steps=24)).to(torch.float32).numpy()
    bias = torch.cos(torch.linspace(0, 3.14159, steps=24)).to(torch.float32).numpy()
    eps = 1e-6
    list_of_inputs.append(copy.deepcopy({
        "input": x, "num_groups": num_groups, "weight": weight, "bias": bias, "eps": eps
    }))

    return list_of_inputs

generated_inputs["torch.nn.functional.group_norm"] = group_norm_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.group_norm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.group_norm'.")


check_valid('torch.nn.functional.group_norm', generated_inputs['torch.nn.functional.group_norm'], lib="torch", suffix=0)
