
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def arcsin_inputs():
    list_of_inputs = []

    t = torch.tensor(0.5, dtype=torch.float32)
    input_np = t.numpy()
    out_np = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_np, "out": out_np}))

    t = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=torch.float64)
    input_np = t.numpy()
    out_np = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_np, "out": out_np}))

    t = torch.tensor([[-2.0, -1.2, -0.8], [0.8, 1.3, 2.0]], dtype=torch.float32)
    input_np = t.numpy()
    out_np = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_np, "out": out_np}))

    t = torch.tensor([[[0.0, 0.25], [0.5, 0.75]], [[-0.1, -0.25], [-0.5, -0.75]]], dtype=torch.float64)
    input_np = t.numpy()
    out_np = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_np, "out": out_np}))

    t = torch.tensor([1+0j, 2+3j, -1-1j, 0+0j], dtype=torch.complex64)
    input_np = t.numpy()
    out_np = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_np, "out": out_np}))

    t = torch.tensor([[0+1j, -2+0.5j], [3-4j, -0.2+0j]], dtype=torch.complex128)
    input_np = t.numpy()
    out_np = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_np, "out": out_np}))

    t = torch.tensor([[[[-1.0, 0.0, 0.7]], [[1.0, -0.7, 0.3]]]], dtype=torch.float32)
    input_np = t.numpy()
    out_np = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_np, "out": out_np}))

    t = torch.tensor([np.nan, np.inf, -np.inf, 0.5, -0.5], dtype=torch.float64)
    input_np = t.numpy()
    out_np = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_np, "out": out_np}))

    t = torch.empty((0,), dtype=torch.float32)
    input_np = t.numpy()
    out_np = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_np, "out": out_np}))

    base = torch.arange(12, dtype=torch.float64).reshape(2, 6)
    t = base[:, ::2].t() / 6.0
    input_np = t.numpy()
    out_np = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_np, "out": out_np}))

    t = torch.tensor([[0.0, -0.25, 0.5], [0.75, -1.0, 1.0], [1.2, -1.5, 0.0]], dtype=torch.float32)
    input_np = t.numpy()
    out_np = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_np, "out": out_np}))

    t = torch.tensor([[[0.0, 0.1], [0.2, 0.3]], [[-0.4, -0.5], [-0.6, -0.7]], [[0.8, 0.9], [1.0, -1.0]]], dtype=torch.float64)
    input_np = t.numpy()
    out_np = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_np, "out": out_np}))

    return list_of_inputs

generated_inputs["torch.arcsin"] = arcsin_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.arcsin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.arcsin'.")


check_valid('torch.arcsin', generated_inputs['torch.arcsin'], lib="torch", suffix=0)
