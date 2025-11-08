
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def square_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 with negatives and zeros
    t = torch.tensor([-2.5, 0.0, 3.5], dtype=torch.float32)
    out_t = torch.empty_like(t)
    input_dict = {
        "input": t.numpy(),
        "out": out_t.numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float64 matrix
    t = torch.tensor([[-1.0, 2.0], [3.0, -4.0]], dtype=torch.float64)
    out_t = torch.empty_like(t)
    input_dict = {
        "input": t.numpy(),
        "out": out_t.numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D int32 tensor
    t = torch.arange(-12, 12, dtype=torch.int32).view(2, 3, 4)
    out_t = torch.empty_like(t)
    input_dict = {
        "input": t.numpy(),
        "out": out_t.numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 0-D scalar float64
    t = torch.tensor(3.14, dtype=torch.float64)
    out_t = torch.empty((), dtype=torch.float64)
    input_dict = {
        "input": t.numpy(),
        "out": out_t.numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D int16 tensor
    t = torch.randint(-10, 10, (2, 2, 2, 3), dtype=torch.int16)
    out_t = torch.empty_like(t)
    input_dict = {
        "input": t.numpy(),
        "out": out_t.numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D uint8 tensor
    t = torch.randint(0, 10, (4, 5), dtype=torch.uint8)
    out_t = torch.empty_like(t)
    input_dict = {
        "input": t.numpy(),
        "out": out_t.numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D complex64 tensor
    real = torch.randn(2, 3, dtype=torch.float32)
    imag = torch.randn(2, 3, dtype=torch.float32)
    t = torch.complex(real, imag)
    out_t = torch.empty_like(t)
    input_dict = {
        "input": t.numpy(),
        "out": out_t.numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D complex128 tensor
    real = torch.tensor([1.0, -2.0, 0.5, -0.75], dtype=torch.float64)
    imag = torch.tensor([0.1, 0.2, -0.3, 0.4], dtype=torch.float64)
    t = torch.complex(real, imag)
    out_t = torch.empty_like(t)
    input_dict = {
        "input": t.numpy(),
        "out": out_t.numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Non-contiguous slice (float32)
    base = torch.arange(1, 13, dtype=torch.float32).view(3, 4)
    t = base[:, ::2]
    out_t = torch.empty_like(t)
    input_dict = {
        "input": t.numpy(),
        "out": out_t.numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: int8 with edge values
    t = torch.tensor([120, -120, 127, -128], dtype=torch.int8)
    out_t = torch.empty_like(t)
    input_dict = {
        "input": t.numpy(),
        "out": out_t.numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: float32 with very large magnitudes
    t = torch.tensor([1e20, -1e10, 3e8], dtype=torch.float32)
    out_t = torch.empty_like(t)
    input_dict = {
        "input": t.numpy(),
        "out": out_t.numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: 5D float16 tensor
    t = torch.randn((1, 2, 1, 2, 3), dtype=torch.float16)
    out_t = torch.empty_like(t)
    input_dict = {
        "input": t.numpy(),
        "out": out_t.numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.square"] = square_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.square' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.square'.")


check_valid('torch.square', generated_inputs['torch.square'], lib="torch", suffix=0)
