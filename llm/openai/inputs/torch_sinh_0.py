
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def sinh_inputs():
    list_of_inputs = []

    input = np.array([0.0, 1.0, -1.0, 3.5], dtype=np.float32)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = np.linspace(-5, 5, 12, dtype=np.float64).reshape(3, 4)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = np.array([-20.0, -10.0, 0.0, 10.0], dtype=np.float32).reshape(2, 1, 1, 2)
    out = np.zeros_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = np.array(-0.5, dtype=np.float64)
    out = np.empty((), dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = np.empty((0, 3), dtype=np.float32)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    base = np.arange(24, dtype=np.float64).reshape(4, 6)
    input = base[:, ::2] * 0.1 - 1.0
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = np.array([1e10, -1e10, 5e5, -5e5], dtype=np.float64)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = (np.ones((1, 2, 1, 2, 1), dtype=np.float32) * -7.25)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = (np.arange(24, dtype=np.float32).reshape(2, 3, 4) - 12) * 0.1
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = np.zeros((5, 5, 5), dtype=np.float64)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    return list_of_inputs

generated_inputs["torch.sinh"] = sinh_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.sinh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sinh'.")


check_valid('torch.sinh', generated_inputs['torch.sinh'], lib="torch", suffix=0)
