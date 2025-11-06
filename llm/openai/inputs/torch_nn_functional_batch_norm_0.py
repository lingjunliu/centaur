
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def batch_norm_inputs():
    list_of_inputs = []

    # Input 1
    inp = torch.randn(4, 3, dtype=torch.float32).numpy()
    C = 3
    running_mean = torch.zeros(C, dtype=torch.float32).numpy()
    running_var = (torch.rand(C, dtype=torch.float32) + 0.5).numpy()
    weight = torch.randn(C, dtype=torch.float32).numpy()
    bias = torch.randn(C, dtype=torch.float32).numpy()
    input_dict = {
        "input": inp,
        "running_mean": running_mean,
        "running_var": running_var,
        "weight": weight,
        "bias": bias,
        "training": False,
        "momentum": 0.1,
        "eps": 1e-5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    inp = torch.randn(2, 4, 5, dtype=torch.float64).numpy()
    C = 4
    running_mean = torch.zeros(C, dtype=torch.float64).numpy()
    running_var = (torch.rand(C, dtype=torch.float64) + 0.5).numpy()
    weight = torch.randn(C, dtype=torch.float64).numpy()
    bias = torch.randn(C, dtype=torch.float64).numpy()
    input_dict = {
        "input": inp,
        "running_mean": running_mean,
        "running_var": running_var,
        "weight": weight,
        "bias": bias,
        "training": True,
        "momentum": 0.05,
        "eps": 1e-3
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    inp = torch.randn(1, 1, 8, 8, dtype=torch.float32).numpy()
    C = 1
    running_mean = torch.zeros(C, dtype=torch.float32).numpy()
    running_var = (torch.rand(C, dtype=torch.float32) + 0.5).numpy()
    weight = torch.ones(C, dtype=torch.float32).numpy()
    bias = torch.zeros(C, dtype=torch.float32).numpy()
    input_dict = {
        "input": inp,
        "running_mean": running_mean,
        "running_var": running_var,
        "weight": weight,
        "bias": bias,
        "training": True,
        "momentum": 0.9,
        "eps": 1e-5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    inp = torch.randn(2, 3, 4, 4, 4, dtype=torch.float32).numpy()
    C = 3
    running_mean = torch.zeros(C, dtype=torch.float32).numpy()
    running_var = (torch.rand(C, dtype=torch.float32) + 0.5).numpy()
    weight = torch.randn(C, dtype=torch.float32).numpy()
    bias = torch.randn(C, dtype=torch.float32).numpy()
    input_dict = {
        "input": inp,
        "running_mean": running_mean,
        "running_var": running_var,
        "weight": weight,
        "bias": bias,
        "training": False,
        "momentum": 0.2,
        "eps": 1e-4
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    inp = torch.randn(3, 2, 6, dtype=torch.float32).numpy()
    C = 2
    running_mean = torch.zeros(C, dtype=torch.float32).numpy()
    running_var = (torch.rand(C, dtype=torch.float32) + 0.5).numpy()
    weight = torch.randn(C, dtype=torch.float32).numpy()
    bias = torch.randn(C, dtype=torch.float32).numpy()
    input_dict = {
        "input": inp,
        "running_mean": running_mean,
        "running_var": running_var,
        "weight": weight,
        "bias": bias,
        "training": True,
        "momentum": 0.1,
        "eps": 1e-3
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    inp = torch.randn(5, 8, 7, 7, dtype=torch.float32).numpy()
    C = 8
    running_mean = torch.zeros(C, dtype=torch.float32).numpy()
    running_var = (torch.rand(C, dtype=torch.float32) + 0.5).numpy()
    weight = (2 * torch.rand(C, dtype=torch.float32) - 1).numpy()
    bias = (2 * torch.rand(C, dtype=torch.float32) - 1).numpy()
    input_dict = {
        "input": inp,
        "running_mean": running_mean,
        "running_var": running_var,
        "weight": weight,
        "bias": bias,
        "training": False,
        "momentum": 0.0,
        "eps": 1e-5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 (fixed: ensure N*spatial > 1 when training)
    inp = torch.randn(2, 5, dtype=torch.float64).numpy()
    C = 5
    running_mean = torch.zeros(C, dtype=torch.float64).numpy()
    running_var = (torch.rand(C, dtype=torch.float64) + 0.5).numpy()
    weight = torch.linspace(-1.0, 1.0, steps=C, dtype=torch.float64).numpy()
    bias = torch.linspace(1.0, -1.0, steps=C, dtype=torch.float64).numpy()
    input_dict = {
        "input": inp,
        "running_mean": running_mean,
        "running_var": running_var,
        "weight": weight,
        "bias": bias,
        "training": True,
        "momentum": 1.0,
        "eps": 1e-6
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    inp = (-3.0 + 2.0 * torch.randn(4, 2, 10, dtype=torch.float32)).numpy()
    C = 2
    running_mean = torch.zeros(C, dtype=torch.float32).numpy()
    running_var = (torch.rand(C, dtype=torch.float32) + 0.5).numpy()
    weight = torch.randn(C, dtype=torch.float32).numpy()
    bias = torch.randn(C, dtype=torch.float32).numpy()
    input_dict = {
        "input": inp,
        "running_mean": running_mean,
        "running_var": running_var,
        "weight": weight,
        "bias": bias,
        "training": False,
        "momentum": 0.3,
        "eps": 1e-3
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    inp = torch.randn(2, 4, 3, 5, dtype=torch.float32).numpy()
    C = 4
    running_mean = torch.zeros(C, dtype=torch.float32).numpy()
    running_var = (torch.rand(C, dtype=torch.float32) + 0.5).numpy()
    weight = torch.ones(C, dtype=torch.float32).numpy()
    bias = (-0.5 * torch.ones(C, dtype=torch.float32)).numpy()
    input_dict = {
        "input": inp,
        "running_mean": running_mean,
        "running_var": running_var,
        "weight": weight,
        "bias": bias,
        "training": True,
        "momentum": 0.7,
        "eps": 1e-4
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    inp = torch.randn(2, 6, 2, 2, 3, dtype=torch.float32).numpy()
    C = 6
    running_mean = torch.zeros(C, dtype=torch.float32).numpy()
    running_var = (torch.rand(C, dtype=torch.float32) + 0.5).numpy()
    weight = (torch.arange(C, dtype=torch.float32) / C).numpy()
    bias = (-torch.arange(C, dtype=torch.float32) / (C + 1)).numpy()
    input_dict = {
        "input": inp,
        "running_mean": running_mean,
        "running_var": running_var,
        "weight": weight,
        "bias": bias,
        "training": False,
        "momentum": 0.5,
        "eps": 1e-3
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    inp = torch.randn(3, 7, 5, 5, dtype=torch.float64).numpy()
    C = 7
    running_mean = torch.zeros(C, dtype=torch.float64).numpy()
    running_var = (torch.rand(C, dtype=torch.float64) + 0.5).numpy()
    weight = torch.randn(C, dtype=torch.float64).numpy()
    bias = torch.zeros(C, dtype=torch.float64).numpy()
    input_dict = {
        "input": inp,
        "running_mean": running_mean,
        "running_var": running_var,
        "weight": weight,
        "bias": bias,
        "training": True,
        "momentum": 0.25,
        "eps": 1e-1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    inp = torch.randn(2, 9, dtype=torch.float32).numpy()
    C = 9
    running_mean = torch.zeros(C, dtype=torch.float32).numpy()
    running_var = (torch.rand(C, dtype=torch.float32) + 0.5).numpy()
    weight = torch.randn(C, dtype=torch.float32).numpy()
    bias = torch.randn(C, dtype=torch.float32).numpy()
    input_dict = {
        "input": inp,
        "running_mean": running_mean,
        "running_var": running_var,
        "weight": weight,
        "bias": bias,
        "training": False,
        "momentum": 0.001,
        "eps": 1e-5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.functional.batch_norm"] = batch_norm_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.batch_norm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.batch_norm'.")


check_valid('torch.nn.functional.batch_norm', generated_inputs['torch.nn.functional.batch_norm'], lib="torch", suffix=0)
