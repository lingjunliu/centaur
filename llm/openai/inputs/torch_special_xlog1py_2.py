
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def xlog1py_inputs():
    list_of_inputs = []

    input = torch.tensor([0.0, 1.0, -1.0], dtype=torch.float32).numpy()
    other = 0.0
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor([[1.0, 2.0], [-3.5, 4.5]], dtype=torch.float64).numpy()
    other = 1.0
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor(3.14159265, dtype=torch.float32).numpy()
    other = 2.0
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    base = torch.arange(24, dtype=torch.float32).reshape(2, 3, 2, 2) - 12.0
    input = base.numpy()
    other = -1.0
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor(
        [[[-1e-6, 1e-6], [1.5, -2.5]],
         [[0.3, -0.3], [10.0, -10.0]]],
        dtype=torch.float64
    ).numpy()
    other = 1e-6
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor([float('nan'), float('inf'), float('-inf'), 5.0], dtype=torch.float32).numpy()
    other = 3.0
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.empty(0, dtype=torch.float64).numpy()
    other = 0.5
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor([-1.0, 0.0, 1.0, 2.0], dtype=torch.float32).reshape(1, 2, 1, 2, 1).numpy()
    other = 1e20
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    base = (torch.arange(12, dtype=torch.float32).reshape(3, 4) - 6.0)
    input = base[:, ::2].numpy()
    other = -2.0
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor([0.0, 1000.0, -1000.0, 1e30, -1e30], dtype=torch.float64).numpy()
    other = 0.75
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.linspace(-5.0, 5.0, steps=11, dtype=torch.float32).reshape(1, 11).numpy()
    other = -0.999999999999
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.randn(2, 3, 4, dtype=torch.float64).numpy()
    other = 0.0
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    return list_of_inputs

generated_inputs["torch.special.xlog1py_2"] = xlog1py_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.special.xlog1py_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.xlog1py_2'.")


check_valid('torch.special.xlog1py', generated_inputs['torch.special.xlog1py_2'], lib="torch", suffix=2)
