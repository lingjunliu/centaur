
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def log10_inputs():
    list_of_inputs = []

    # 1
    input = torch.tensor([0.1, 1.0, 10.0, 100.0], dtype=torch.float32).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # 2
    input = torch.tensor([[1.0, 2.0], [10.0, 1000.0]], dtype=torch.float64).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # 3
    input = torch.tensor([-1.0, 0.0, 0.5, -100.0], dtype=torch.float32).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # 4
    input = torch.linspace(-5.0, 5.0, steps=24, dtype=torch.float32).reshape(2, 3, 4).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # 5
    input = torch.tensor(10.0, dtype=torch.float64).numpy()
    out = np.empty((), dtype=input.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # 6
    input = torch.empty(0, dtype=torch.float32).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # 7 (non-contiguous via transpose)
    base = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    input = base.t().numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # 8 (float16)
    input = torch.tensor([[-0.1, 0.1], [1.0, 10.0], [100.0, -10.0], [0.001, 0.01]], dtype=torch.float16).reshape(2, 2, 2).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # 9 (4D, out aliased to input)
    input = (torch.ones(2, 1, 3, 4, dtype=torch.float32) * 10.0).numpy()
    out = input
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # 10 (complex64)
    input = torch.tensor([1+1j, -1+2j, -3j, 10+0j], dtype=torch.complex64).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # 11 (non-finite values)
    input = torch.tensor([float('inf'), float('-inf'), float('nan'), 1.0], dtype=torch.float32).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # 12 (very small positives, float64)
    input = torch.tensor([[1e-30, 1e-100], [1e-5, 1e-10]], dtype=torch.float64).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    return list_of_inputs

generated_inputs["torch.log10"] = log10_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.log10' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.log10'.")


check_valid('torch.log10', generated_inputs['torch.log10'], lib="torch", suffix=0)
