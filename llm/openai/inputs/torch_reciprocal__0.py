
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def reciprocal__inputs():
    list_of_inputs = []

    input = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[1.0, 0.0, -3.0], [4.0, 5.0, -0.25]], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[[1.0, -2.0, 0.3], [0.0, 4.0, -5.5]],
                          [[-0.125, 2.5, 10.0], [1.5, -3.0, 0.0]]], dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    arr = torch.arange(12, dtype=torch.float32).reshape(1, 2, 2, 3)
    arr = arr - 6.0
    input = arr.numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    t = torch.arange(12, dtype=torch.float64).reshape(3, 4)
    input = t.t().numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([1+2j, -0.5+0j, 0+1j], dtype=torch.complex64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[0+0j, -2-3j], [4+0j, 1-1j]], dtype=torch.complex128).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = np.array([1e-8, -1e-10, 3e-12], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = np.array([np.inf, -np.inf, np.nan, 1.0, -1.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input}))

    base = np.arange(20, dtype=np.float64)
    input = base[::3]
    list_of_inputs.append(copy.deepcopy({"input": input}))

    real = torch.tensor([[[1.0, -2.0, 0.0]],
                         [[3.5, -0.1, 2.2]]], dtype=torch.float32)
    imag = torch.tensor([[[0.5, 1.5, -3.0]],
                         [[0.0, -2.0, 4.0]]], dtype=torch.float32)
    input = (real + 1j * imag).to(torch.complex64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    return list_of_inputs

generated_inputs["torch.reciprocal_"] = reciprocal__inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.reciprocal_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.reciprocal_'.")


check_valid('torch.reciprocal_', generated_inputs['torch.reciprocal_'], lib="torch", suffix=0)
