
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def i0_inputs():
    list_of_inputs = []

    t = torch.tensor([0.0, 1.0, 2.5, 5.0], dtype=torch.float32)
    input = t.numpy()
    out = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    t = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=torch.float32)
    input = t.numpy()
    out = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    t = torch.tensor([[0.0, 1.0, 2.0],
                      [3.5, -4.0, -0.5]], dtype=torch.float64)
    input = t.numpy()
    out = torch.zeros_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    t = torch.tensor([[[-1.0, -0.1, 0.0],
                       [0.1, 1.0, 2.0]],
                      [[3.0, -2.5, 4.0],
                       [5.0, 6.0, -7.0]]], dtype=torch.float16)
    input = t.numpy()
    out = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    t = torch.tensor([-50.0, -10.0, 0.0, 10.0, 50.0, 100.0], dtype=torch.float64)
    input = t.numpy()
    out = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    t = torch.arange(0, 6, dtype=torch.float32)
    input = t.numpy()
    out = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    t = torch.tensor(3.141592653589793, dtype=torch.float64)
    input = t.numpy()
    out = torch.empty((), dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    t_base = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    t = t_base.t()
    input = t.numpy()
    out = torch.zeros_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    t = torch.linspace(-2.0, 2.0, steps=24, dtype=torch.float32).reshape(2, 1, 3, 4)
    input = t.numpy()
    out = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    t = torch.tensor([[0.0, 1.0, 2.0],
                      [3.0, 4.0, 5.0]], dtype=torch.float64)
    input = t.numpy()
    out = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    t = torch.tensor([-1e-4, -1e-5, 0.0, 1e-5, 1e-4], dtype=torch.float16)
    input = t.numpy()
    out = torch.zeros_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    t = torch.empty((0,), dtype=torch.float32)
    input = t.numpy()
    out = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    return list_of_inputs

generated_inputs["torch.special.i0"] = i0_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.special.i0' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.i0'.")


check_valid('torch.special.i0', generated_inputs['torch.special.i0'], lib="torch", suffix=0)
