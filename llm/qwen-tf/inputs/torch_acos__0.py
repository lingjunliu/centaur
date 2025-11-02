
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def acos__inputs():
    list_of_inputs = []

    input = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = (torch.rand((2, 3), dtype=torch.float64) * 2 - 1).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor(0.3, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[[-1.0, -0.25], [0.25, 1.0]]], dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = np.linspace(-1, 1, num=10, dtype=np.float32)[::2]
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = np.asfortranarray(np.array([[-0.8, -0.3, 0.0], [0.2, 0.7, 1.0]], dtype=np.float64))
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([1 + 1j, -0.2 + 0.3j, 0.5 - 0.5j], dtype=torch.complex64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[0.1 + 2j, -1.0 + 0.0j], [0.0 - 1.5j, 0.8 + 0.9j]], dtype=torch.complex128).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = (torch.rand((2, 1, 2, 3), dtype=torch.float32) * 2 - 1).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = np.array([-1.5, -1.0001, 1.0001, 2.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = np.array([np.nan, np.inf, -np.inf, 0.5, -0.5], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input}))

    base = np.linspace(-1, 1, num=7, dtype=np.float64)
    input = base[::-1]
    list_of_inputs.append(copy.deepcopy({"input": input}))

    return list_of_inputs

generated_inputs["torch.acos_"] = acos__inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.acos_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.acos_'.")

check_valid('torch.acos_', generated_inputs['torch.acos_'], lib="torch", suffix=0)
