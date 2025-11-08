
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def log1p_inputs():
    list_of_inputs = []

    input = torch.tensor([0.0, 1.0, 2.5], dtype=torch.float32).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = torch.tensor([-0.999, -0.5, 0.0, 3.0], dtype=torch.float64).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = torch.zeros((2, 3), dtype=torch.float32).numpy()
    out = np.zeros_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = torch.tensor([[[1e-6, 1.0], [10.0, 1e5]],
                          [[-0.1, -0.9], [2.0, 5.0]]], dtype=torch.float64).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = torch.tensor(3.14, dtype=torch.float64).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = torch.empty(0, dtype=torch.float64).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    base = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    input = base[:, ::2].numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = torch.randn(2, 1, 3, 4, dtype=torch.float32).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = torch.tensor([-0.5 + 0.1j, 0.0 + 0.0j, 2.0 - 3.0j], dtype=torch.complex64).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = torch.tensor([[-1.0 + 0.0j, 1.0 + 2.0j],
                          [3.0 - 4.0j, -0.1 + 0.1j]], dtype=torch.complex128).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = torch.linspace(-0.9999, 1000.0, steps=17, dtype=torch.float64).reshape(1, 17).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = torch.empty((2, 0, 3), dtype=torch.float32).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    return list_of_inputs

generated_inputs["torch.special.log1p"] = log1p_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.special.log1p' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.log1p'.")


check_valid('torch.special.log1p', generated_inputs['torch.special.log1p'], lib="torch", suffix=0)
