
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def slogdet_inputs():
    list_of_inputs = []

    input = torch.tensor([[4.0, 1.0],
                          [2.0, 3.0]], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.diag(torch.tensor([1.0, 1.0, -2.0], dtype=torch.float64)).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[1.0, 2.0],
                          [2.0, 4.0]], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.randn(4, 2, 2, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[1+2j, 3-1j],
                          [0+4j, -2+0j]], dtype=torch.complex64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    real = torch.randn(3, 3, 3, dtype=torch.float64)
    imag = torch.randn(3, 3, 3, dtype=torch.float64)
    input = (real + 1j * imag).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    A = torch.randn(5, 5, dtype=torch.float64)
    input = (A.transpose(-2, -1) @ A + 1e-3 * torch.eye(5, dtype=torch.float64)).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    A = torch.randn(2, 4, 4, dtype=torch.float32)
    input = (A.transpose(-2, -1) @ A + 0.05 * torch.eye(4, dtype=torch.float32)).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[1.0, 1.0],
                          [1.0, 1.0 + 1e-8]], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = (1e8 * torch.tensor([[1.0, -2.0, 3.0],
                                 [0.5, 4.0, -1.5],
                                 [2.0, 0.0, 1.0]], dtype=torch.float32)).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[-3.5]], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    C = torch.tensor([[0.0, 2.0, 1.0],
                      [3.5, -1.2, 4.8],
                      [7.1, 0.0, -2.2]], dtype=torch.float64).numpy()
    input = C.T
    list_of_inputs.append(copy.deepcopy({"input": input}))

    return list_of_inputs

generated_inputs["torch.slogdet"] = slogdet_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.slogdet' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.slogdet'.")


check_valid('torch.slogdet', generated_inputs['torch.slogdet'], lib="torch", suffix=0)
