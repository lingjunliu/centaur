
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def expit_inputs():
    list_of_inputs = []

    input_t = torch.tensor([1.0, 2.0, 3.0, -1.0, -2.0, 0.0], dtype=torch.float32)
    input = input_t.numpy()
    out = torch.empty_like(input_t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input_t = torch.tensor([[-100.0, -1.0, 0.0],
                            [1.0, 10.0, 100.0]], dtype=torch.float64)
    input = input_t.numpy()
    out = torch.empty_like(input_t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input_t = torch.tensor(5.0, dtype=torch.float32)
    input = input_t.numpy()
    out = torch.empty((), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input_t = torch.linspace(-8, 8, steps=24, dtype=torch.float16).reshape(2, 3, 4)
    input = input_t.numpy()
    out = torch.empty_like(input_t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input_t = torch.empty((0,), dtype=torch.float32)
    input = input_t.numpy()
    out = torch.empty_like(input_t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input_t = torch.empty((2, 0, 3), dtype=torch.float64)
    input = input_t.numpy()
    out = torch.empty_like(input_t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input_t = torch.tensor([float('inf'), float('-inf'), float('nan'), -0.0, 0.0], dtype=torch.float64)
    input = input_t.numpy()
    out = torch.empty_like(input_t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    base = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    input_t = base.t()
    input = input_t.numpy()
    out = torch.empty_like(input_t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    torch.manual_seed(0)
    input_t = (torch.randn(1, 2, 3, 4, dtype=torch.float32) * 5.0)
    input = input_t.numpy()
    out = torch.empty_like(input_t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input_t = torch.tensor([-12.0, -2.5, 0.0, 2.5, 12.0], dtype=torch.float32)
    input = input_t.numpy()
    out = input
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input_t = torch.linspace(-20, 20, steps=20, dtype=torch.float32)[::2]
    input = input_t.numpy()
    out = torch.empty_like(input_t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input_t = torch.tensor([-20.0, -10.0, -1.0, 0.0, 1.0, 10.0, 20.0], dtype=torch.float16)
    input = input_t.numpy()
    out = torch.empty_like(input_t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input_t = torch.tensor([1000.0, -1000.0], dtype=torch.float64)
    input = input_t.numpy()
    out = torch.empty_like(input_t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    return list_of_inputs

generated_inputs["torch.special.expit"] = expit_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.special.expit' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.expit'.")


check_valid('torch.special.expit', generated_inputs['torch.special.expit'], lib="torch", suffix=0)
