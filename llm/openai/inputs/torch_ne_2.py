
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np, math

def ne_inputs():
    list_of_inputs = []

    input = torch.tensor([1, 2, 3, 2, -1], dtype=torch.int64).numpy()
    other = 2.0
    out = torch.empty(input.shape, dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor([[-1.5, 0.0, 2.5], [3.0, -4.0, 5.0]], dtype=torch.float32).numpy()
    other = -1.0
    out = torch.empty(input.shape, dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor([[[0, 1, 2], [3, 4, 5]]], dtype=torch.int16).numpy()
    other = 0.0
    out = torch.empty(input.shape, dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor(3.14, dtype=torch.float64).numpy()
    other = 3.140001
    out = torch.empty((), dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor([[True, False, True], [False, False, True]], dtype=torch.bool).numpy()
    other = 1.0
    out = torch.empty(input.shape, dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor([0, 127, 255], dtype=torch.uint8).numpy()
    other = 255.0
    out = torch.empty(input.shape, dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor([float('nan'), 1.0, -float('inf'), float('inf')], dtype=torch.float64).numpy()
    other = float('nan')
    out = torch.empty(input.shape, dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor([1+0j, 0+1j, 0+0j, -2+3j], dtype=torch.complex64).numpy()
    other = 0.0
    out = torch.empty(input.shape, dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.linspace(-1, 1, 12, dtype=torch.float32).reshape(2, 1, 3, 2).numpy()
    other = 0.5
    out = torch.empty(input.shape, dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor([-0.0, 0.0, 1.0, -1.0], dtype=torch.float32).numpy()
    other = 0.0
    out = torch.empty(input.shape, dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.empty((0, 3), dtype=torch.float32).numpy()
    other = -0.0
    out = torch.empty(input.shape, dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.arange(-10, 10, dtype=torch.int32).reshape(4, 5).numpy()
    other = -1.0
    out = torch.empty(input.shape, dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor(np.arange(0, 48).reshape(2, 2, 2, 3, 2), dtype=torch.float16).numpy()
    other = 0.0
    out = torch.empty(input.shape, dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    return list_of_inputs

generated_inputs["torch.ne_2"] = ne_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.ne_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.ne_2'.")


check_valid('torch.ne', generated_inputs['torch.ne_2'], lib="torch", suffix=2)
