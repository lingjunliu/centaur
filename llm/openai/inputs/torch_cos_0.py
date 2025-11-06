
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def cos_inputs():
    list_of_inputs = []

    t = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=torch.float32)
    input_dict = {
        "input": t.numpy(),
        "out": torch.empty_like(t).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = torch.randn(2, 3, dtype=torch.float64)
    input_dict = {
        "input": t.numpy(),
        "out": torch.empty_like(t).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = torch.linspace(-3.14, 3.14, steps=8, dtype=torch.float16).reshape(2, 2, 2)
    input_dict = {
        "input": t.numpy(),
        "out": torch.empty_like(t).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = torch.tensor([1+1j, -2+0.5j, -0.0-1j], dtype=torch.complex64)
    input_dict = {
        "input": t.numpy(),
        "out": torch.empty_like(t).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = torch.randn(2, 2, dtype=torch.float64) + 1j * torch.randn(2, 2, dtype=torch.float64)
    input_dict = {
        "input": t.numpy(),
        "out": torch.empty_like(t).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = torch.empty(0, dtype=torch.float32)
    input_dict = {
        "input": t.numpy(),
        "out": torch.empty_like(t).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4)
    input_dict = {
        "input": t.numpy(),
        "out": torch.empty_like(t).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    base = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    t = base.t()
    input_dict = {
        "input": t.numpy(),
        "out": torch.empty_like(t).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = torch.tensor([float('nan'), float('inf'), float('-inf'), float(torch.pi)], dtype=torch.float32)
    input_dict = {
        "input": t.numpy(),
        "out": torch.empty_like(t).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = torch.tensor([1e20, -1e20, 1e-20, -1e-20], dtype=torch.float64)
    input_dict = {
        "input": t.numpy(),
        "out": torch.empty_like(t).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = torch.arange(-10, 10, dtype=torch.float32)[::3]
    input_dict = {
        "input": t.numpy(),
        "out": torch.empty_like(t).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = torch.empty(2, 0, 3, dtype=torch.float32)
    input_dict = {
        "input": t.numpy(),
        "out": torch.empty_like(t).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.cos"] = cos_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.cos' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.cos'.")


check_valid('torch.cos', generated_inputs['torch.cos'], lib="torch", suffix=0)
