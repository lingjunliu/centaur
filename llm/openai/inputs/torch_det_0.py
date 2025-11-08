
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, numpy as np, copy

def det_inputs():
    list_of_inputs = []

    input = torch.tensor([[3.5]], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[1.0, 2.0],
                          [3.0, 4.0]], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[0.0, -2.0, 5.0],
                          [3.0, 1.0, -4.0],
                          [-1.0, 2.0, 3.0]], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[4.0, 0.0, -2.0, 1.0],
                          [3.0, 5.0, 1.0, 0.0],
                          [2.0, -1.0, 3.0, 4.0],
                          [0.0, 2.0, -3.0, 6.0]], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[1.0 + 2.0j, -3.0 + 0.5j],
                          [2.0 - 1.0j, 4.0 + 0.0j]], dtype=torch.complex64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[0.0 + 1.0j, 2.0 - 3.0j, -1.0 + 0.0j],
                          [4.0 + 0.0j, -2.0 + 2.0j, 1.0 - 1.0j],
                          [3.0 - 2.0j, 0.0 + 0.0j, 5.0 + 4.0j]], dtype=torch.complex128).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.randn((5, 2, 2), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[[1.0, -2.0, 3.0],
                           [0.0, 4.0, -1.0],
                           [2.0, 1.0, 0.0]],
                          [[-1.0, 0.0, 2.0],
                           [3.0, -4.0, 1.0],
                           [5.0, 2.0, -3.0]]], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.randn((2, 2, 2, 2), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[2.0, 4.0, 6.0],
                          [1.0, 2.0, 3.0],
                          [0.0, 0.0, 0.0]], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    theta = 0.7
    rot = torch.tensor([[np.cos(theta), -np.sin(theta), 0.0],
                        [np.sin(theta),  np.cos(theta), 0.0],
                        [0.0,            0.0,          1.0]], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": rot}))

    diag_vals = torch.tensor([10.0, -2.0, 0.5, 3.0], dtype=torch.float64)
    input = torch.diag(diag_vals).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    return list_of_inputs

generated_inputs["torch.det"] = det_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.det' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.det'.")


check_valid('torch.det', generated_inputs['torch.det'], lib="torch", suffix=0)
