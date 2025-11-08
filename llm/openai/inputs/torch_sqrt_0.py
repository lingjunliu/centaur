
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def sqrt_inputs():
    list_of_inputs = []

    t = torch.tensor([4.0, 1.0, -1.0, 0.0], dtype=torch.float32)
    input = t.numpy()
    out = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    t = torch.tensor([[0.25, 100.0], [1e-8, 1e16]], dtype=torch.float64)
    input = t.numpy()
    out = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    t = torch.tensor([[-2.0, 0.5], [3.0, 10.0]], dtype=torch.float16)
    input = t.numpy()
    out = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    t = torch.tensor(9.0, dtype=torch.float32)
    input = t.numpy()
    out = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    t = torch.randn(2, 3, 4, dtype=torch.float32)
    input = t.numpy()
    out = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    t = torch.arange(1.0, 10.0, dtype=torch.float32).view(3, 3).t()
    input = t.numpy()
    out = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    t = torch.tensor([0.0, -0.0, 16.0, -9.0], dtype=torch.float32)
    input = t.numpy()
    out = torch.zeros_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    t = torch.tensor([1+0j, -1+0j, 3+4j, 0+0j], dtype=torch.complex64)
    input = t.numpy()
    out = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    t = torch.tensor([[0+1j, -4+0j], [16+0j, -9+12j]], dtype=torch.complex128)
    input = t.numpy()
    out = torch.zeros_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    t = torch.rand(2, 2, 2, 3, dtype=torch.float32)
    input = t.numpy()
    out = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    t = torch.tensor([float('nan'), float('inf'), -float('inf'), 16.0], dtype=torch.float32)
    input = t.numpy()
    out = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    base = torch.linspace(0.0, 9.0, steps=10, dtype=torch.float64).view(2, 5)
    t = base[:, ::2]
    input = t.numpy()
    out = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    return list_of_inputs

generated_inputs["torch.sqrt"] = sqrt_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.sqrt' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sqrt'.")


check_valid('torch.sqrt', generated_inputs['torch.sqrt'], lib="torch", suffix=0)
