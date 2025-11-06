
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def clip_grad_norm__2_inputs():
    list_of_inputs = []

    parameters = torch.tensor([1.0, -2.0, 3.5], dtype=torch.float32).numpy()
    max_norm = 1.0
    norm_type = 2.0
    error_if_nonfinite = False
    input_dict = {
        "parameters": parameters,
        "max_norm": max_norm,
        "norm_type": norm_type,
        "error_if_nonfinite": error_if_nonfinite
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    parameters = (torch.ones((2, 3), dtype=torch.float64) * 5).numpy()
    max_norm = 0.5
    norm_type = float('inf')
    error_if_nonfinite = True
    input_dict = {
        "parameters": parameters,
        "max_norm": max_norm,
        "norm_type": norm_type,
        "error_if_nonfinite": error_if_nonfinite
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    parameters = torch.randn((4, 4), dtype=torch.float16).numpy()
    max_norm = 10.0
    norm_type = 1.0
    error_if_nonfinite = False
    input_dict = {
        "parameters": parameters,
        "max_norm": max_norm,
        "norm_type": norm_type,
        "error_if_nonfinite": error_if_nonfinite
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    parameters = torch.tensor(3.14, dtype=torch.float32).numpy()
    max_norm = 3.0
    norm_type = 2.0
    error_if_nonfinite = False
    input_dict = {
        "parameters": parameters,
        "max_norm": max_norm,
        "norm_type": norm_type,
        "error_if_nonfinite": error_if_nonfinite
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    parameters = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).add_(-12).numpy()
    max_norm = 5.0
    norm_type = 2.0
    error_if_nonfinite = True
    input_dict = {
        "parameters": parameters,
        "max_norm": max_norm,
        "norm_type": norm_type,
        "error_if_nonfinite": error_if_nonfinite
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    parameters = torch.empty((0,), dtype=torch.float32).numpy()
    max_norm = 1.0
    norm_type = 2.0
    error_if_nonfinite = False
    input_dict = {
        "parameters": parameters,
        "max_norm": max_norm,
        "norm_type": norm_type,
        "error_if_nonfinite": error_if_nonfinite
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    parameters = torch.randn(1, 2, 3, 1, 2, dtype=torch.float32).numpy()
    max_norm = 2.5
    norm_type = 2.0
    error_if_nonfinite = False
    input_dict = {
        "parameters": parameters,
        "max_norm": max_norm,
        "norm_type": norm_type,
        "error_if_nonfinite": error_if_nonfinite
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    parameters = torch.linspace(-1, 1, steps=7, dtype=torch.float64).numpy()
    max_norm = 0.1
    norm_type = 1.0
    error_if_nonfinite = True
    input_dict = {
        "parameters": parameters,
        "max_norm": max_norm,
        "norm_type": norm_type,
        "error_if_nonfinite": error_if_nonfinite
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    parameters = torch.randn(10, dtype=torch.float32).numpy()
    max_norm = 100.0
    norm_type = 4.0
    error_if_nonfinite = False
    input_dict = {
        "parameters": parameters,
        "max_norm": max_norm,
        "norm_type": norm_type,
        "error_if_nonfinite": error_if_nonfinite
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    parameters = torch.randn(3, 3, dtype=torch.float32).numpy()
    max_norm = 1e-6
    norm_type = 2.0
    error_if_nonfinite = False
    input_dict = {
        "parameters": parameters,
        "max_norm": max_norm,
        "norm_type": norm_type,
        "error_if_nonfinite": error_if_nonfinite
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    parameters = torch.tensor([float('nan'), float('inf'), -float('inf')], dtype=torch.float32).numpy()
    max_norm = 1.0
    norm_type = 2.0
    error_if_nonfinite = False
    input_dict = {
        "parameters": parameters,
        "max_norm": max_norm,
        "norm_type": norm_type,
        "error_if_nonfinite": error_if_nonfinite
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    parameters = torch.randn(64, 128, dtype=torch.float32).numpy()
    max_norm = 3.0
    norm_type = 3.0
    error_if_nonfinite = True
    input_dict = {
        "parameters": parameters,
        "max_norm": max_norm,
        "norm_type": norm_type,
        "error_if_nonfinite": error_if_nonfinite
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.utils.clip_grad_norm__2"] = clip_grad_norm__2_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.utils.clip_grad_norm__2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.utils.clip_grad_norm__2'.")


check_valid('torch.nn.utils.clip_grad_norm_', generated_inputs['torch.nn.utils.clip_grad_norm__2'], lib="torch", suffix=2)
