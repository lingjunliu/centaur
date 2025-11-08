
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def erf_inputs():
    list_of_inputs = []

    t = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=torch.float32)
    input = t.numpy()
    out = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    t = torch.tensor([[0.0, 0.5, 1.0],
                      [1.5, 2.0, -2.5]], dtype=torch.float64)
    input = t.numpy()
    out = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    t = torch.tensor(1.2345, dtype=torch.float32)
    input = t.numpy()
    out = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    t = torch.tensor([[[0.1, -0.2],
                       [3.0, -4.0]],
                      [[5.5, -6.5],
                       [7.75, -8.25]]], dtype=torch.float16)
    input = t.numpy()
    out = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    t = torch.tensor([float('nan'), float('inf'), -float('inf'), 0.0, -0.0], dtype=torch.float64)
    input = t.numpy()
    out = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    t = torch.tensor([1e-6, -1e-6, 5.0, -5.0, 10.0, -10.0], dtype=torch.float32)
    input = t.numpy()
    out = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    t = torch.empty((0,), dtype=torch.float32)
    input = t.numpy()
    out = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    t = (torch.arange(2*3*4*5, dtype=torch.float32).reshape(2, 3, 4, 5) / 10.0) - 5.0
    input = t.numpy()
    out = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    t = torch.tensor([[-0.1, -0.2, -0.3],
                      [-1.0, -2.0, -3.0]], dtype=torch.float64)
    input = t.numpy()
    out = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    t = torch.tensor([0.0, 0.0625, -0.125, 0.25, -0.5], dtype=torch.float16)
    input = t.numpy()
    out = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    t = torch.empty((2, 0, 3), dtype=torch.float32)
    input = t.numpy()
    out = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    base = torch.arange(20, dtype=torch.float32).view(4, 5)
    t = base[:, ::2]
    input = t.numpy()
    out = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    return list_of_inputs

generated_inputs["torch.erf"] = erf_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.erf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.erf'.")


check_valid('torch.erf', generated_inputs['torch.erf'], lib="torch", suffix=0)
