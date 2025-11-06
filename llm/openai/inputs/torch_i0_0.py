
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, numpy as np, copy

def i0_inputs():
    list_of_inputs = []

    input = np.array([0.0, 1.0, -1.0, 3.5], dtype=np.float32)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = np.array([[0.0, 2.0, -2.0], [5.0, -5.0, 10.0]], dtype=np.float64)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = np.array(2.5, dtype=np.float32)
    out = np.empty((), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = np.random.uniform(-3, 3, size=(2, 3, 4)).astype(np.float64)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = np.array([], dtype=np.float64)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = np.array([20.0, 30.0, 50.0], dtype=np.float64)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = np.array([[-1.0, -0.5, -10.0], [1.0, 0.5, 10.0]], dtype=np.float32)
    out = np.zeros_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    base = np.arange(12, dtype=np.float64).reshape(3, 4)
    input = base.T
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = np.random.randn(2, 2, 2, 3).astype(np.float64)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = np.array([np.nan, np.inf, -np.inf, 0.0, 1.0, -1.0], dtype=np.float32)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = np.zeros((1, 1, 5), dtype=np.float32)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = np.linspace(-8.0, 8.0, num=17, dtype=np.float64).reshape(1, 17)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    return list_of_inputs

generated_inputs["torch.i0"] = i0_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.i0' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.i0'.")


check_valid('torch.i0', generated_inputs['torch.i0'], lib="torch", suffix=0)
