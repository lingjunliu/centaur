
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def lgamma_inputs():
    list_of_inputs = []

    inp_t = torch.tensor([0.5, 1.0, 1.5, 2.5, 10.0], dtype=torch.float64)
    out_t = torch.empty_like(inp_t, dtype=inp_t.dtype)
    list_of_inputs.append(copy.deepcopy({"input": inp_t.numpy(), "out": out_t.numpy()}))

    inp_t = torch.tensor([[-3.0, -2.0, -1.0],
                          [0.0, 0.5, 1.0]], dtype=torch.float32)
    out_t = torch.empty_like(inp_t, dtype=inp_t.dtype)
    list_of_inputs.append(copy.deepcopy({"input": inp_t.numpy(), "out": out_t.numpy()}))

    inp_t = torch.tensor([[[-3, -2, -1],
                           [0, 1, 2]],
                          [[-4, -5, 3],
                           [4, 5, 6]]], dtype=torch.int64)
    out_t = torch.empty_like(inp_t, dtype=torch.float64)
    list_of_inputs.append(copy.deepcopy({"input": inp_t.numpy(), "out": out_t.numpy()}))

    inp_t = torch.tensor(3.1415926, dtype=torch.float32)
    out_t = torch.empty_like(inp_t, dtype=inp_t.dtype)
    list_of_inputs.append(copy.deepcopy({"input": inp_t.numpy(), "out": out_t.numpy()}))

    inp_t = torch.tensor([20.0, 50.0, 100.0, 170.0], dtype=torch.float64)
    out_t = torch.empty_like(inp_t, dtype=inp_t.dtype)
    list_of_inputs.append(copy.deepcopy({"input": inp_t.numpy(), "out": out_t.numpy()}))

    inp_t = torch.tensor([float('inf'), float('-inf'), float('nan'), -0.5, 0.1], dtype=torch.float64)
    out_t = torch.empty_like(inp_t, dtype=inp_t.dtype)
    list_of_inputs.append(copy.deepcopy({"input": inp_t.numpy(), "out": out_t.numpy()}))

    inp_t = torch.linspace(-2.5, 2.5, steps=6, dtype=torch.float32).reshape(2, 1, 3, 1)
    out_t = torch.empty_like(inp_t, dtype=inp_t.dtype)
    list_of_inputs.append(copy.deepcopy({"input": inp_t.numpy(), "out": out_t.numpy()}))

    inp_t = torch.tensor([-1.0 + 1e-7, -2.0 - 1e-6, -3.0 + 1e-9, -4.0 - 1e-9], dtype=torch.float64)
    out_t = torch.empty_like(inp_t, dtype=inp_t.dtype)
    list_of_inputs.append(copy.deepcopy({"input": inp_t.numpy(), "out": out_t.numpy()}))

    inp_t = torch.tensor([1e-8, 1e-12, 1e-20], dtype=torch.float64)
    out_t = torch.empty_like(inp_t, dtype=inp_t.dtype)
    list_of_inputs.append(copy.deepcopy({"input": inp_t.numpy(), "out": out_t.numpy()}))

    inp_t = torch.tensor([[-0.1, -0.9, -1.1],
                          [2.0, 5.5, 7.2]], dtype=torch.float32)
    out_t = torch.empty_like(inp_t, dtype=inp_t.dtype)
    list_of_inputs.append(copy.deepcopy({"input": inp_t.numpy(), "out": out_t.numpy()}))

    inp_t = torch.tensor([-10, -5, 0], dtype=torch.int32)
    out_t = torch.empty_like(inp_t, dtype=torch.float64)
    list_of_inputs.append(copy.deepcopy({"input": inp_t.numpy(), "out": out_t.numpy()}))

    inp_t = torch.tensor([-20.5, -30.25, -100.125], dtype=torch.float64)
    out_t = torch.empty_like(inp_t, dtype=inp_t.dtype)
    list_of_inputs.append(copy.deepcopy({"input": inp_t.numpy(), "out": out_t.numpy()}))

    return list_of_inputs

generated_inputs["torch.lgamma"] = lgamma_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.lgamma' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.lgamma'.")


check_valid('torch.lgamma', generated_inputs['torch.lgamma'], lib="torch", suffix=0)
