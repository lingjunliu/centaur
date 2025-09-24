
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def ndtri_inputs():
    list_of_inputs = []

    inp_t = torch.tensor(0.5, dtype=torch.float32)
    input = inp_t.numpy()
    out = torch.empty_like(inp_t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    inp_t = torch.tensor([1e-10, 0.2, 0.5, 0.8, 1 - 1e-10], dtype=torch.float64)
    input = inp_t.numpy()
    out = torch.empty_like(inp_t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    inp_t = torch.tensor([[1e-3, 1e-2, 0.1],
                          [0.9, 0.99, 0.999]], dtype=torch.float32)
    input = inp_t.numpy()
    out = torch.empty_like(inp_t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    inp_t = torch.rand((2, 1, 3), dtype=torch.float32)
    input = inp_t.numpy()
    out = torch.empty_like(inp_t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    inp_t = torch.tensor([0.0, 1.0], dtype=torch.float64)
    input = inp_t.numpy()
    out = torch.empty_like(inp_t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    inp_t = torch.tensor([-0.1, 1.2, 2.0, -5.0], dtype=torch.float32)
    input = inp_t.numpy()
    out = torch.empty_like(inp_t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    inp_t = torch.empty((0,), dtype=torch.float32)
    input = inp_t.numpy()
    out = torch.empty_like(inp_t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    base = torch.linspace(0.0, 1.0, steps=10, dtype=torch.float64)
    inp_t = base[::2]
    input = inp_t.numpy()
    out = torch.empty_like(inp_t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    inp_t = torch.tensor([[float('nan'), 0.3],
                          [0.7, float('nan')]], dtype=torch.float32)
    input = inp_t.numpy()
    out = torch.empty_like(inp_t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    inp_t = torch.rand((2, 2, 1, 3), dtype=torch.float64)
    input = inp_t.numpy()
    out = torch.empty_like(inp_t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    inp_t = torch.rand((4, 3), dtype=torch.float32).t()
    input = inp_t.numpy()
    out = torch.empty_like(inp_t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    return list_of_inputs

generated_inputs["torch.special.ndtri"] = ndtri_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.special.ndtri' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.ndtri'.")


check_valid('torch.special.ndtri', generated_inputs['torch.special.ndtri'], lib="torch", suffix=0)
