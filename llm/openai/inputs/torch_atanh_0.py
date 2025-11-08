
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def atanh_inputs():
    list_of_inputs = []

    input = np.array([-0.5, 0.0, 0.5], dtype=np.float32)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = np.array([[0.2, -0.3, 0.7],
                      [-0.8, 0.1, -0.6]], dtype=np.float64)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = np.array([-1.0, -0.9999, 0.9999, 1.0], dtype=np.float32)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = np.array([-1.5, -1.1, 1.2, 2.0], dtype=np.float64)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = np.empty((0,), dtype=np.float32)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = np.array([[[-0.1, 0.2, -0.3, 0.4]],
                      [[0.05, -0.25, 0.35, -0.45]]], dtype=np.float16)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = np.linspace(-0.8, 0.8, num=24, dtype=np.float32).reshape(1, 2, 3, 4)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = np.array([0.5 + 0.5j, -0.2 + 0.7j, 0.1 - 0.3j], dtype=np.complex64)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = np.array([[0.3 + 0.2j, -0.4 - 0.1j],
                      [0.0 + 0.0j, -0.9 + 0.9j]], dtype=np.complex128)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = np.array([np.nan, 0.25, -0.25, np.inf, -np.inf], dtype=np.float32)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    eps = np.nextafter(1.0, 0.0) - 1.0
    near_one = 1.0 + eps
    input = np.array([-near_one, near_one], dtype=np.float64)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = (np.random.RandomState(0).uniform(-0.95, 0.95, size=(5,)) ).astype(np.float32)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    return list_of_inputs

generated_inputs["torch.atanh"] = atanh_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.atanh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.atanh'.")


check_valid('torch.atanh', generated_inputs['torch.atanh'], lib="torch", suffix=0)
