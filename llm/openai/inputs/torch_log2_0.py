
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def log2_inputs():
    list_of_inputs = []

    input = torch.rand(5, dtype=torch.float32).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = (torch.rand(2, 3, dtype=torch.float64) * 10 + 1e-3).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = (torch.rand(2, 3, 4, dtype=torch.float16) + 1e-2).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = torch.tensor([-8.0, -1.0, -0.0, 0.0, 0.5, 1.0, 2.0, 4.0], dtype=torch.float32).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = torch.linspace(-10.0, -0.1, steps=7, dtype=torch.float32).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = torch.tensor([0.0, -0.0, float('inf'), float('-inf'), float('nan')], dtype=torch.float64).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = torch.tensor([1+1j, -1+2j, 0+1j, 2+0j, -3-4j], dtype=torch.complex64).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    a = torch.randn(2, 2, dtype=torch.float64)
    b = torch.randn(2, 2, dtype=torch.float64)
    input = torch.complex(a, b).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = torch.tensor([], dtype=torch.float32).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = torch.arange(24, dtype=torch.float32).view(2, 3, 4)[:, :, ::2].numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = (torch.rand(2, 2, 2, 2, 2, dtype=torch.float32) + 0.5).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = torch.tensor([2.0**10, 2.0**20, 2.0**-5, 1e10, 1e-10], dtype=torch.float64).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    return list_of_inputs

generated_inputs["torch.log2"] = log2_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.log2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.log2'.")


check_valid('torch.log2', generated_inputs['torch.log2'], lib="torch", suffix=0)
