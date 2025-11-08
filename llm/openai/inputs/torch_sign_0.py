
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def sign_inputs():
    list_of_inputs = []

    input = torch.tensor([0.7, -1.2, 0.0, 2.3], dtype=torch.float32).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = torch.tensor([[1.0, -2.5, 0.0], [3.3, -4.4, 5.5]], dtype=torch.float64).numpy()
    out = np.zeros_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = torch.tensor(0.0, dtype=torch.float32).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = torch.tensor([[[-1, 0, 1], [2, -3, 0]]], dtype=torch.int32).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = torch.tensor([-128, -1, 0, 1, 127], dtype=torch.int8).numpy()
    out = np.zeros_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = torch.tensor([[0.0, -0.0, 1e-7, -1e-7], [1e3, -1e3, 0.0, -0.0]], dtype=torch.float16).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = torch.arange(-12, 12, dtype=torch.int64).reshape(1, 2, 3, 4).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = torch.tensor([float('nan'), float('inf'), float('-inf'), -0.0, 0.0, 2.0, -3.0], dtype=torch.float32).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = torch.empty((0, 3), dtype=torch.float32).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    base = torch.arange(12, dtype=torch.float32).reshape(3, 4).numpy()
    input = base[::2, ::-1].copy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = (torch.arange(24, dtype=torch.float32) - 12).reshape(1, 2, 1, 3, 4).numpy()
    out = np.zeros_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    return list_of_inputs

generated_inputs["torch.sign"] = sign_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.sign' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sign'.")


check_valid('torch.sign', generated_inputs['torch.sign'], lib="torch", suffix=0)
