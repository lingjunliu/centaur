
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def sin_inputs():
    list_of_inputs = []

    inp = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.1415927], dtype=torch.float32).numpy()
    out = np.empty_like(inp)
    list_of_inputs.append(copy.deepcopy({"input": inp, "out": out}))

    inp = torch.tensor([[-0.5, 0.0, 0.5],
                        [1.5, -2.5, 10.0]], dtype=torch.float64).numpy()
    out = np.zeros_like(inp)
    list_of_inputs.append(copy.deepcopy({"input": inp, "out": out}))

    inp = torch.tensor(0.12345, dtype=torch.float32).numpy()
    out = np.empty_like(inp)
    list_of_inputs.append(copy.deepcopy({"input": inp, "out": out}))

    inp = (torch.arange(12, dtype=torch.float16).reshape(2, 2, 3) / 10).numpy()
    out = np.zeros_like(inp)
    list_of_inputs.append(copy.deepcopy({"input": inp, "out": out}))

    inp = (torch.ones((2, 1, 2, 3), dtype=torch.float32) * 0.25).numpy()
    out = np.empty_like(inp)
    list_of_inputs.append(copy.deepcopy({"input": inp, "out": out}))

    inp = torch.tensor([1+2j, -3-4j, 0+1j, -2+0j], dtype=torch.complex64).numpy()
    out = np.empty_like(inp)
    list_of_inputs.append(copy.deepcopy({"input": inp, "out": out}))

    inp = torch.tensor([[np.pi + 0j, 0 - 1j],
                        [2 + 3j, -5 + 2j]], dtype=torch.complex128).numpy()
    out = np.zeros_like(inp)
    list_of_inputs.append(copy.deepcopy({"input": inp, "out": out}))

    inp = torch.tensor([-2*np.pi, -np.pi, -np.pi/2, 0.0, np.pi/2, np.pi, 2*np.pi],
                       dtype=torch.float32).numpy()
    out = np.empty_like(inp)
    list_of_inputs.append(copy.deepcopy({"input": inp, "out": out}))

    inp = torch.tensor([np.nan, np.inf, -np.inf, -0.0, 0.0], dtype=torch.float64).numpy()
    out = np.empty_like(inp)
    list_of_inputs.append(copy.deepcopy({"input": inp, "out": out}))

    base = torch.arange(12, dtype=torch.float32).reshape(3, 4).numpy()
    inp = base.T
    out = np.empty_like(inp)
    list_of_inputs.append(copy.deepcopy({"input": inp, "out": out}))

    base = torch.linspace(-6.0, 6.0, steps=9, dtype=torch.float64).numpy()
    inp = base[::-1]
    out = np.zeros_like(inp)
    list_of_inputs.append(copy.deepcopy({"input": inp, "out": out}))

    inp = torch.empty(0, dtype=torch.float32).numpy()
    out = np.empty_like(inp)
    list_of_inputs.append(copy.deepcopy({"input": inp, "out": out}))

    return list_of_inputs

generated_inputs["torch.sin"] = sin_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.sin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sin'.")


check_valid('torch.sin', generated_inputs['torch.sin'], lib="torch", suffix=0)
