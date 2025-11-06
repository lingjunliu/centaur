
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def neg_inputs():
    list_of_inputs = []

    # Input 1: 1D float32
    input = torch.tensor([1.0, -2.5, 0.0, 3.14], dtype=torch.float32).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 2: 2D float64
    input = torch.tensor([[1.0, -1.0], [2.5, -3.5]], dtype=torch.float64).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 3: 2D int32
    input = torch.tensor([[1, -2, 3], [-4, 5, -6]], dtype=torch.int32).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 4: 3D int64
    input = torch.tensor([[[1, -1], [2, -2]], [[-3, 3], [4, -4]]], dtype=torch.int64).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 5: 1D complex64
    input = torch.tensor([1+2j, -3+0.5j, 0-1j], dtype=torch.complex64).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 6: 2D complex128
    input = torch.tensor([[1-1j, -2+2j], [3+0j, -4-4j]], dtype=torch.complex128).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 7: 3D float16
    input = torch.randn(2, 2, 3, dtype=torch.float16).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 8: 0-D scalar float32
    input = torch.tensor(3.5, dtype=torch.float32).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 9: Empty 1D float32
    input = torch.tensor([], dtype=torch.float32).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 10: 4D float32
    input = torch.arange(2*3*4*5, dtype=torch.float32).reshape(2, 3, 4, 5).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 11: Non-contiguous slice float64
    base = torch.arange(24, dtype=torch.float64).reshape(4, 6).numpy()
    input = base[:, ::2]
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 12: 1D int16 with negatives
    input = torch.tensor([327, -128, 0, 45, -67], dtype=torch.int16).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    return list_of_inputs

generated_inputs["torch.neg"] = neg_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.neg' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.neg'.")


check_valid('torch.neg', generated_inputs['torch.neg'], lib="torch", suffix=0)
