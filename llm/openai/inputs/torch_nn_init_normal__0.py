
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, numpy as np, copy

def nn_init_normal_inputs():
    list_of_inputs = []

    tensor = torch.zeros(5, dtype=torch.float32).numpy()
    mean = 0.0
    std = 1.0
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "mean": mean, "std": std}))

    tensor = torch.ones((2, 3), dtype=torch.float64).numpy()
    mean = 5.0
    std = 2.5
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "mean": mean, "std": std}))

    tensor = torch.empty((2, 2, 2), dtype=torch.float16).numpy()
    mean = -1.5
    std = 0.1
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "mean": mean, "std": std}))

    tensor = torch.tensor(0.0, dtype=torch.float32).numpy()
    mean = 10.0
    std = 0.0
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "mean": mean, "std": std}))

    tensor = torch.empty((0, 3), dtype=torch.float32).numpy()
    mean = 0.0
    std = 1.0
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "mean": mean, "std": std}))

    tensor = torch.zeros((2, 3, 4, 5), dtype=torch.float32).numpy()
    mean = 0.3
    std = 0.7
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "mean": mean, "std": std}))

    base = torch.arange(12, dtype=torch.float64).reshape(3, 4)
    tensor = base.t().numpy()
    mean = -3.14
    std = 2.0
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "mean": mean, "std": std}))

    tensor = torch.zeros((3,), dtype=torch.complex64).numpy()
    mean = 0.0
    std = 1.0
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "mean": mean, "std": std}))

    tensor = torch.zeros((2, 2), dtype=torch.complex128).numpy()
    mean = -2.0
    std = 3.0
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "mean": mean, "std": std}))

    tensor = torch.empty((1024,), dtype=torch.float16).numpy()
    mean = 0.0
    std = 1e-3
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "mean": mean, "std": std}))

    tensor = torch.zeros((1, 2, 3, 1, 4), dtype=torch.float32).numpy()
    mean = 2.0
    std = 0.5
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "mean": mean, "std": std}))

    tensor = torch.empty((3, 0, 2), dtype=torch.float32).numpy()
    mean = -100.0
    std = 10.0
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "mean": mean, "std": std}))

    return list_of_inputs

generated_inputs["torch.nn.init.normal_"] = nn_init_normal_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.init.normal_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.init.normal_'.")


check_valid('torch.nn.init.normal_', generated_inputs['torch.nn.init.normal_'], lib="torch", suffix=0)
