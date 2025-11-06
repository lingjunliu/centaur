
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def vander_inputs():
    list_of_inputs = []

    x = torch.tensor([1, 2, 3, 5], dtype=torch.int64).numpy()
    N = 4
    increasing = False
    input_dict = {"x": x, "N": N, "increasing": increasing}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32).numpy()
    N = 3
    increasing = True
    input_dict = {"x": x, "N": N, "increasing": increasing}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = torch.tensor([-3, -1, 0, 2, 4], dtype=torch.int32).numpy()
    N = 6
    increasing = False
    input_dict = {"x": x, "N": N, "increasing": increasing}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = torch.tensor([0.5, -1.5, 2.25], dtype=torch.float64).numpy()
    N = 2
    increasing = True
    input_dict = {"x": x, "N": N, "increasing": increasing}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = torch.tensor([1+2j, -3+0.5j, 0-1j], dtype=torch.complex64).numpy()
    N = 3
    increasing = True
    input_dict = {"x": x, "N": N, "increasing": increasing}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = torch.tensor([2-1j], dtype=torch.complex128).numpy()
    N = 4
    increasing = False
    input_dict = {"x": x, "N": N, "increasing": increasing}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = torch.tensor([10_000, -20_000, 30_000], dtype=torch.int64).numpy()
    N = 2
    increasing = True
    input_dict = {"x": x, "N": N, "increasing": increasing}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float16).numpy()
    N = 3
    increasing = True
    input_dict = {"x": x, "N": N, "increasing": increasing}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64).numpy()
    N = 5
    increasing = False
    input_dict = {"x": x, "N": N, "increasing": increasing}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = torch.linspace(-1, 1, steps=7, dtype=torch.float32).numpy()
    N = 4
    increasing = True
    input_dict = {"x": x, "N": N, "increasing": increasing}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = torch.tensor([], dtype=torch.float32).numpy()
    N = 3
    increasing = True
    input_dict = {"x": x, "N": N, "increasing": increasing}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = torch.randn(5, dtype=torch.float64).numpy()
    N = 5
    increasing = False
    input_dict = {"x": x, "N": N, "increasing": increasing}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.vander"] = vander_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.vander' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.vander'.")


check_valid('torch.vander', generated_inputs['torch.vander'], lib="torch", suffix=0)
