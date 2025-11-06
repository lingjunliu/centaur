
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def softplus_inputs():
    list_of_inputs = []

    input = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32).numpy()
    beta = 1.0
    threshold = 20.0
    list_of_inputs.append(copy.deepcopy({"input": input, "beta": beta, "threshold": threshold}))

    input = torch.tensor([[1.0, -2.0, 3.0],
                          [0.5, -0.5, 0.0]], dtype=torch.float64).numpy()
    beta = 0.5
    threshold = 15.0
    list_of_inputs.append(copy.deepcopy({"input": input, "beta": beta, "threshold": threshold}))

    input = torch.tensor([[[1.0, -1.0], [2.0, -2.0]],
                          [[3.0, -3.0], [4.0, -4.0]]], dtype=torch.float16).numpy()
    beta = 2.0
    threshold = 10.0
    list_of_inputs.append(copy.deepcopy({"input": input, "beta": beta, "threshold": threshold}))

    input = torch.tensor(3.0, dtype=torch.float32).numpy()
    beta = 1.0
    threshold = 0.0
    list_of_inputs.append(copy.deepcopy({"input": input, "beta": beta, "threshold": threshold}))

    input = torch.tensor([50.0, 100.0, 1000.0], dtype=torch.float32).numpy()
    beta = 1.0
    threshold = 20.0
    list_of_inputs.append(copy.deepcopy({"input": input, "beta": beta, "threshold": threshold}))

    input = torch.tensor([-50.0, -100.0, -1000.0], dtype=torch.float32).numpy()
    beta = 1.5
    threshold = 20.0
    list_of_inputs.append(copy.deepcopy({"input": input, "beta": beta, "threshold": threshold}))

    input = torch.tensor([float('inf'), float('-inf'), float('nan')], dtype=torch.float32).numpy()
    beta = 1.0
    threshold = 20.0
    list_of_inputs.append(copy.deepcopy({"input": input, "beta": beta, "threshold": threshold}))

    input = torch.randn(1, 3, 2, 2, dtype=torch.float32).numpy()
    beta = 5.0
    threshold = 50.0
    list_of_inputs.append(copy.deepcopy({"input": input, "beta": beta, "threshold": threshold}))

    input = torch.tensor([-1e-6, 0.0, 1e-6], dtype=torch.float64).numpy()
    beta = 100.0
    threshold = 50.0
    list_of_inputs.append(copy.deepcopy({"input": input, "beta": beta, "threshold": threshold}))

    input = torch.tensor([[-10.0, -1.0, -0.1, 0.1, 1.0]], dtype=torch.float16).numpy()
    beta = 10.0
    threshold = 1.0
    list_of_inputs.append(copy.deepcopy({"input": input, "beta": beta, "threshold": threshold}))

    input = torch.linspace(-5, 5, steps=20, dtype=torch.float32).reshape(4, 5).numpy()
    beta = 0.1
    threshold = 2.0
    list_of_inputs.append(copy.deepcopy({"input": input, "beta": beta, "threshold": threshold}))

    input = torch.empty((0,), dtype=torch.float32).numpy()
    beta = 1.0
    threshold = 20.0
    list_of_inputs.append(copy.deepcopy({"input": input, "beta": beta, "threshold": threshold}))

    return list_of_inputs

generated_inputs["torch.nn.functional.softplus"] = softplus_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.softplus' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.softplus'.")


check_valid('torch.nn.functional.softplus', generated_inputs['torch.nn.functional.softplus'], lib="torch", suffix=0)
