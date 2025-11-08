
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def acosh_inputs():
    list_of_inputs = []

    input = torch.linspace(1.0, 3.0, steps=5, dtype=torch.float32).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = torch.tensor([[1.0, 2.0, 10.0],
                          [1.5, 3.0, 4.0]], dtype=torch.float64).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = torch.tensor(1.0, dtype=torch.float32).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = torch.tensor([[[0.5, 1.0, 2.0],
                           [0.9, 1.1, 5.0]]], dtype=torch.float64).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    base = torch.arange(1, 25, dtype=torch.float64).reshape(4, 6)
    input = base[:, ::2].numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = torch.tensor([1+0j, 0.5+2j, -3+4j, 2-0.1j], dtype=torch.complex64).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = torch.tensor([[1+1j, 2-3j],
                          [10+0j, -0.5+0.5j]], dtype=torch.complex128).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = (torch.ones((2, 3), dtype=torch.float16) * 1.25).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = (torch.rand((2, 1, 2, 3), dtype=torch.float32) * 2.0 + 1.0).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = torch.empty((0, 3), dtype=torch.float32).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = torch.tensor([float('inf'), 1.0, 2.0, float('nan')], dtype=torch.float64).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    mat = torch.tensor([[1.0, 1.1, 1.2],
                        [5.0, 6.0, 7.0],
                        [100.0, 1e6, 1e12]], dtype=torch.float64)
    input = mat.t().numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    return list_of_inputs

generated_inputs["torch.acosh"] = acosh_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.acosh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.acosh'.")


check_valid('torch.acosh', generated_inputs['torch.acosh'], lib="torch", suffix=0)
