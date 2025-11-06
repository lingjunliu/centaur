
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def floor_inputs():
    list_of_inputs = []

    inp = torch.tensor([1.5, -2.3, 0.0, 3.9], dtype=torch.float32).numpy()
    out = np.empty_like(inp)
    list_of_inputs.append(copy.deepcopy({"input": inp, "out": out}))

    inp = torch.randn(2, 3, dtype=torch.float64).numpy()
    out = np.empty_like(inp)
    list_of_inputs.append(copy.deepcopy({"input": inp, "out": out}))

    inp = torch.tensor([-1.1, -0.9, 0.1, 1.1, 2.9, -2.1, 3.5, -3.5], dtype=torch.float16).reshape(2, 2, 2).numpy()
    out = np.empty_like(inp)
    list_of_inputs.append(copy.deepcopy({"input": inp, "out": out}))

    inp = torch.tensor([1, -2, 3, -4], dtype=torch.int64).numpy()
    out = np.empty_like(inp)
    list_of_inputs.append(copy.deepcopy({"input": inp, "out": out}))

    inp = torch.arange(-6, 6, dtype=torch.int32).reshape(3, 4).numpy()
    out = np.empty_like(inp)
    list_of_inputs.append(copy.deepcopy({"input": inp, "out": out}))

    inp = torch.randn(3, 4, dtype=torch.float32).t().numpy()
    out = np.empty_like(inp)
    list_of_inputs.append(copy.deepcopy({"input": inp, "out": out}))

    inp = torch.empty(0, dtype=torch.float32).numpy()
    out = np.empty_like(inp)
    list_of_inputs.append(copy.deepcopy({"input": inp, "out": out}))

    inp = torch.tensor([float('nan'), float('inf'), float('-inf'), -0.0, 0.0, 1.999], dtype=torch.float64).numpy()
    out = np.empty_like(inp)
    list_of_inputs.append(copy.deepcopy({"input": inp, "out": out}))

    inp = torch.tensor([1e10 + 0.9, -1e10 - 0.1], dtype=torch.float64).numpy()
    out = np.empty_like(inp)
    list_of_inputs.append(copy.deepcopy({"input": inp, "out": out}))

    inp = torch.randn(2, 1, 3, 2, dtype=torch.float32).numpy()
    out = np.empty_like(inp)
    list_of_inputs.append(copy.deepcopy({"input": inp, "out": out}))

    inp = torch.tensor([0, 1, 255], dtype=torch.uint8).numpy()
    out = np.empty_like(inp)
    list_of_inputs.append(copy.deepcopy({"input": inp, "out": out}))

    inp = torch.tensor([-123, 0, 456, -789, 1024, -2048], dtype=torch.int16).reshape(1, 2, 3).numpy()
    out = np.empty_like(inp)
    list_of_inputs.append(copy.deepcopy({"input": inp, "out": out}))

    return list_of_inputs

generated_inputs["torch.floor"] = floor_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.floor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.floor'.")


check_valid('torch.floor', generated_inputs['torch.floor'], lib="torch", suffix=0)
