
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def psi_inputs():
    list_of_inputs = []

    input_t = torch.tensor([0.1, 1.0, 2.5, 10.0], dtype=torch.float32)
    input = input_t.numpy()
    out = torch.empty_like(input_t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input_t = torch.tensor([[-0.5, -1.5, 0.0],
                            [0.5, 1.5, 50.0]], dtype=torch.float64)
    input = input_t.numpy()
    out = torch.empty_like(input_t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input_t = torch.tensor([[[0.1, 0.2],
                             [1.0, 2.0]],
                            [[3.0, 4.0],
                             [5.0, 6.5]]], dtype=torch.float32)
    input = input_t.numpy()
    out = torch.empty_like(input_t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input_t = torch.tensor(3.141592653589793, dtype=torch.float64)
    input = input_t.numpy()
    out = torch.empty_like(input_t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    base = torch.arange(24, dtype=torch.float32).reshape(2, 1, 3, 4)
    input_t = (base + 0.5) / 10.0
    input = input_t.numpy()
    out = torch.empty_like(input_t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input_t = torch.tensor([1e-12, 1e-6, 1e2, 1e6, 1e12], dtype=torch.float64)
    input = input_t.numpy()
    out = torch.empty_like(input_t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input_t = torch.tensor([float('nan'), float('inf'), float('-inf'), -1.0, 0.0, 1.0], dtype=torch.float64)
    input = input_t.numpy()
    out = torch.empty_like(input_t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input_t = torch.tensor([-0.1, -0.9, -1.1, -2.5, -3.7], dtype=torch.float32)
    input = input_t.numpy()
    out = torch.empty_like(input_t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input_t = torch.linspace(0.1, 5.0, steps=12, dtype=torch.float32).reshape(3, 4)
    input = input_t.numpy()
    out = torch.empty_like(input_t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input_t = torch.randn(2, 3, 4, dtype=torch.float64) * 3.0
    input = input_t.numpy()
    out = torch.empty_like(input_t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input_t = torch.tensor([[-1.0 + 1e-6, -2.0 - 1e-6, -3.0 + 1e-12, -4.0 - 1e-12],
                            [-10.0 + 1e-8, -1e-6, 1e-6, 20.0]], dtype=torch.float64)
    input = input_t.numpy()
    out = torch.empty_like(input_t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    return list_of_inputs

generated_inputs["torch.special.psi"] = psi_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.special.psi' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.psi'.")


check_valid('torch.special.psi', generated_inputs['torch.special.psi'], lib="torch", suffix=0)
