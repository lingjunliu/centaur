
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def sparse__inputs():
    list_of_inputs = []

    t = torch.zeros((3, 4), dtype=torch.float32).numpy()
    input_dict = {"tensor": t, "sparsity": 0.5, "std": 0.01}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = torch.ones((10, 10), dtype=torch.float64).numpy()
    input_dict = {"tensor": t, "sparsity": 0.1, "std": 0.05}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = torch.randn((8, 3), dtype=torch.float16).numpy()
    input_dict = {"tensor": t, "sparsity": 0.8, "std": 0.001}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = torch.tensor([[1.0, -2.0, 3.0, -4.0, 5.0]], dtype=torch.float32).numpy()
    input_dict = {"tensor": t, "sparsity": 0.0, "std": 0.02}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = torch.tensor([[-1.0], [2.0], [-3.0], [4.0], [-5.0]], dtype=torch.float32).numpy()
    input_dict = {"tensor": t, "sparsity": 1.0, "std": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = torch.zeros((100, 50), dtype=torch.float32).numpy()
    input_dict = {"tensor": t, "sparsity": 0.95, "std": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = torch.tensor([[0.5, -0.5], [1.5, -1.5]], dtype=torch.float32).numpy()
    input_dict = {"tensor": t, "sparsity": 0.25, "std": 0.1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = torch.full((7, 7), 0.3, dtype=torch.float32).numpy()
    input_dict = {"tensor": t, "sparsity": 0.9, "std": 0.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    base = torch.arange(3 * 7, dtype=torch.float64).view(3, 7).numpy()
    t = base
    input_dict = {"tensor": t, "sparsity": 0.6, "std": 0.05}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = torch.randn((64, 16), dtype=torch.float16).numpy()
    input_dict = {"tensor": t, "sparsity": 0.7, "std": 0.2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = torch.linspace(-1, 1, steps=16, dtype=torch.float32).view(2, 8).numpy()
    input_dict = {"tensor": t, "sparsity": 0.99, "std": 0.0001}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = torch.eye(32, dtype=torch.float64).numpy()
    input_dict = {"tensor": t, "sparsity": 0.33, "std": 3.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.init.sparse_"] = sparse__inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.init.sparse_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.init.sparse_'.")


check_valid('torch.nn.init.sparse_', generated_inputs['torch.nn.init.sparse_'], lib="torch", suffix=0)
