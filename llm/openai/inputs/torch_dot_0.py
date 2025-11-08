
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def dot_inputs():
    list_of_inputs = []

    v1 = torch.tensor([1.0, -2.0, 3.5], dtype=torch.float32).numpy()
    v2 = torch.tensor([4.0, 0.5, -1.0], dtype=torch.float32).numpy()
    out = torch.zeros((), dtype=torch.float32).numpy()
    input_dict = {"input": v1, "tensor": v2, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    v1 = torch.tensor([0.0, 1.0], dtype=torch.float64).numpy()
    v2 = torch.tensor([2.0, -3.5], dtype=torch.float64).numpy()
    out = torch.zeros((), dtype=torch.float64).numpy()
    input_dict = {"input": v1, "tensor": v2, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    v1 = torch.tensor([1, 2, 3, 4], dtype=torch.int64).numpy()
    v2 = torch.tensor([-1, 0, 10, -5], dtype=torch.int64).numpy()
    out = torch.zeros((), dtype=torch.int64).numpy()
    input_dict = {"input": v1, "tensor": v2, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    v1 = torch.tensor([-1000, 2000, -3000], dtype=torch.int32).numpy()
    v2 = torch.tensor([3, -2, 1], dtype=torch.int32).numpy()
    out = torch.zeros((), dtype=torch.int32).numpy()
    input_dict = {"input": v1, "tensor": v2, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    v1 = torch.tensor([300, -400, 500, -600, 700], dtype=torch.int16).numpy()
    v2 = torch.tensor([-1, -1, 1, 1, -1], dtype=torch.int16).numpy()
    out = torch.zeros((), dtype=torch.int16).numpy()
    input_dict = {"input": v1, "tensor": v2, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    v1 = torch.tensor([1.5, -2.25, 0.75, 3.0], dtype=torch.float16).numpy()
    v2 = torch.tensor([-1.0, 2.0, -0.5, 1.0], dtype=torch.float16).numpy()
    out = torch.zeros((), dtype=torch.float16).numpy()
    input_dict = {"input": v1, "tensor": v2, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    v1 = torch.randn(100, dtype=torch.float32).mul_(0.1).numpy()
    v2 = torch.randn(100, dtype=torch.float32).mul_(10.0).numpy()
    out = torch.zeros((), dtype=torch.float32).numpy()
    input_dict = {"input": v1, "tensor": v2, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    v1 = torch.tensor([-7], dtype=torch.int64).numpy()
    v2 = torch.tensor([8], dtype=torch.int64).numpy()
    out = torch.zeros((), dtype=torch.int64).numpy()
    input_dict = {"input": v1, "tensor": v2, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    v1 = torch.tensor([10, -20, 30, -40, 50, -60, 70, -80, 90, -100], dtype=torch.int32).numpy()
    v2 = torch.tensor([-1, 2, -3, 4, -5, 6, -7, 8, -9, 10], dtype=torch.int32).numpy()
    out = torch.zeros((), dtype=torch.int32).numpy()
    input_dict = {"input": v1, "tensor": v2, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    v1 = torch.tensor([1e-12, -2e-12, 3e-12, -4e-12], dtype=torch.float64).numpy()
    v2 = torch.tensor([5e12, -6e12, 7e12, -8e12], dtype=torch.float64).numpy()
    out = torch.zeros((), dtype=torch.float64).numpy()
    input_dict = {"input": v1, "tensor": v2, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    v1 = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32).numpy()
    v2 = torch.tensor([1.0, -2.0, 3.0, -4.0], dtype=torch.float32).numpy()
    out = torch.zeros((), dtype=torch.float32).numpy()
    input_dict = {"input": v1, "tensor": v2, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    v1 = torch.linspace(-5, 5, steps=7, dtype=torch.float32).numpy()
    v2 = torch.linspace(3, -3, steps=7, dtype=torch.float32).numpy()
    out = torch.zeros((), dtype=torch.float32).numpy()
    input_dict = {"input": v1, "tensor": v2, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.dot"] = dot_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.dot' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.dot'.")


check_valid('torch.dot', generated_inputs['torch.dot'], lib="torch", suffix=0)
