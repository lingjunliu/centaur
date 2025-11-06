
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def unsqueeze_inputs():
    list_of_inputs = []

    # Input 1
    input_arr = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32).numpy()
    dim = 0
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim}))

    # Input 2
    input_arr = torch.tensor([1, 0, 1, 0], dtype=torch.int8).numpy()
    dim = 1
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim}))

    # Input 3
    input_arr = torch.arange(6, dtype=torch.float32).reshape(2, 3).numpy()
    dim = -3
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim}))

    # Input 4
    input_arr = torch.tensor(5, dtype=torch.int64).numpy()
    dim = np.int64(0)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim}))

    # Input 5
    input_arr = torch.tensor(3.14, dtype=torch.float64).numpy()
    dim = -1
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim}))

    # Input 6
    input_arr = torch.tensor([[True, False], [False, True]], dtype=torch.bool).numpy()
    dim = np.int32(2)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim}))

    # Input 7
    input_arr = torch.zeros((2, 0, 3), dtype=torch.float32).numpy()
    dim = 3
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim}))

    # Input 8
    input_arr = torch.randn(4, 5, 6, dtype=torch.float64).numpy()
    dim = -4
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim}))

    # Input 9
    input_arr = torch.ones((1, 2, 3, 4), dtype=torch.int32).numpy()
    dim = 2
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim}))

    # Input 10
    input_arr = torch.arange(12, dtype=torch.int64).reshape(3, 4).T.numpy()
    dim = -1
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim}))

    # Input 11
    complex_tensor = (torch.randn(3, dtype=torch.float32) + 1j * torch.randn(3, dtype=torch.float32)).to(torch.complex64)
    input_arr = complex_tensor.numpy()
    dim = 1
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim}))

    # Input 12
    input_arr = torch.randn(2, 1, 1, 3, 4, dtype=torch.float16).numpy()
    dim = -6
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim}))

    # Input 13
    input_arr = torch.empty((0,), dtype=torch.float32).numpy()
    dim = -2
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim}))

    # Input 14
    input_arr = torch.linspace(0, 1, steps=6, dtype=torch.float32).reshape(2, 3).numpy()
    dim = 1
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim}))

    # Input 15
    input_arr = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4).numpy()
    dim = 0
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim}))

    return list_of_inputs

generated_inputs["torch.unsqueeze"] = unsqueeze_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.unsqueeze' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.unsqueeze'.")


check_valid('torch.unsqueeze', generated_inputs['torch.unsqueeze'], lib="torch", suffix=0)
