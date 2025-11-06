
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def atan2_inputs():
    list_of_inputs = []

    input = torch.tensor([0.9041, 0.0196, -0.3108, -2.4423], dtype=torch.float32).numpy()
    other = torch.tensor([1.0, -2.0, 3.0, -4.0], dtype=torch.float32).numpy()
    out = torch.empty((4,), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor([[1.0, -2.0, 3.0],
                          [-4.0, 5.0, -6.0]], dtype=torch.float64).numpy()
    other = torch.tensor([[0.5, -1.5, 2.5],
                          [-3.5, 4.5, -5.5]], dtype=torch.float64).numpy()
    out = torch.empty((2, 3), dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float32).numpy()
    other = torch.tensor([[1.0, 2.0, -3.0],
                          [-4.0, 5.0, 6.0]], dtype=torch.float32).numpy()
    out = torch.empty((2, 3), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor([[[1.0], [-2.0], [3.0]]], dtype=torch.float64).numpy()  # (1,3,1)
    other = torch.tensor([[[0.5, -1.5, 2.5, -3.5]],
                          [[4.5, -5.5, 6.5, -7.5]]], dtype=torch.float64).numpy()  # (2,1,4)
    out = torch.empty((2, 3, 4), dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor(0.5, dtype=torch.float32).numpy()
    other = torch.tensor([1.0, -1.0, 0.0], dtype=torch.float32).numpy()
    out = torch.empty((3,), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor(-1.25, dtype=torch.float64).numpy()
    other = torch.tensor(2.75, dtype=torch.float64).numpy()
    out = torch.empty((), dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor([-1.0, 0.0, 1.0, -2.0, 2.0], dtype=torch.float64).numpy()
    other = torch.tensor([0.0, 0.0, 0.0, 1.0, -1.0], dtype=torch.float64).numpy()
    out = torch.empty((5,), dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor([[1.0, -1.0],
                          [2.0, -2.0]], dtype=torch.float32).numpy()
    other = torch.tensor([[3.0, -3.0],
                          [-4.0, 4.0]], dtype=torch.float64).numpy()
    out = torch.empty((2, 2), dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor([[1.0], [-2.0], [3.0], [-4.0]], dtype=torch.float16).numpy()  # (4,1)
    other = torch.tensor([[0.5, -1.5, 2.5, -3.5]], dtype=torch.float16).numpy()       # (1,4)
    out = torch.empty((4, 4), dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor([[[[1.0], [-1.0]],
                           [[2.0], [-2.0]]],
                          [[[3.0], [-3.0]],
                           [[4.0], [-4.0]]]], dtype=torch.float32).numpy()  # (2,2,2,1)
    other = torch.tensor([[[[1.0, -1.0, 2.0, -2.0, 3.0]]],
                          [[[4.0, -4.0, 5.0, -5.0, 6.0]]]], dtype=torch.float32).numpy()  # (2,1,1,5)
    out = torch.empty((2, 2, 2, 5), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor([-0.0, 0.0, -0.0], dtype=torch.float64).numpy()
    other = torch.tensor([-1.0, 1.0, -2.0], dtype=torch.float64).numpy()
    out = torch.empty((3,), dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor([[1e-45, -1e-45],
                          [1e-38, -1e-38]], dtype=torch.float32).numpy()
    other = torch.tensor([[1e-45, 0.0],
                          [-1e-38, 1e-38]], dtype=torch.float32).numpy()
    out = torch.empty((2, 2), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    return list_of_inputs

generated_inputs["torch.atan2"] = atan2_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.atan2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.atan2'.")


check_valid('torch.atan2', generated_inputs['torch.atan2'], lib="torch", suffix=0)
