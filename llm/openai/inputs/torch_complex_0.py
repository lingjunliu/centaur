
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def complex_inputs():
    list_of_inputs = []

    # Input 1: 1D float32
    real = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32).numpy()
    imag = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32).numpy()
    out = torch.empty(real.shape, dtype=torch.complex64).numpy()
    list_of_inputs.append(copy.deepcopy({"real": real, "imag": imag, "out": out}))

    # Input 2: 2D float32 with negatives
    real = torch.tensor([[1.5, -2.5, 3.0],
                         [-4.0, 0.0, 6.5]], dtype=torch.float32).numpy()
    imag = torch.tensor([[0.5, 2.0, -3.5],
                         [4.5, -1.0, 0.0]], dtype=torch.float32).numpy()
    out = torch.empty(real.shape, dtype=torch.complex64).numpy()
    list_of_inputs.append(copy.deepcopy({"real": real, "imag": imag, "out": out}))

    # Input 3: 3D float32
    real = torch.tensor([[[1.0, -1.0],
                          [2.0, -2.0]],
                         [[3.0, -3.0],
                          [4.0, -4.0]]], dtype=torch.float32).numpy()
    imag = torch.tensor([[[0.1, 0.2],
                          [0.3, 0.4]],
                         [[-0.5, -0.6],
                          [0.7, 0.8]]], dtype=torch.float32).numpy()
    out = torch.empty(real.shape, dtype=torch.complex64).numpy()
    list_of_inputs.append(copy.deepcopy({"real": real, "imag": imag, "out": out}))

    # Input 4: scalar float32
    real = torch.tensor(2.5, dtype=torch.float32).numpy()
    imag = torch.tensor(-1.5, dtype=torch.float32).numpy()
    out = torch.empty((), dtype=torch.complex64).numpy()
    list_of_inputs.append(copy.deepcopy({"real": real, "imag": imag, "out": out}))

    # Input 5: 1D float64
    real = torch.tensor([10.0, 20.0, -30.0, 40.0], dtype=torch.float64).numpy()
    imag = torch.tensor([0.0, -1.0, 2.0, -3.0], dtype=torch.float64).numpy()
    out = torch.empty(real.shape, dtype=torch.complex128).numpy()
    list_of_inputs.append(copy.deepcopy({"real": real, "imag": imag, "out": out}))

    # Input 6: 2D float64 with broadcasting-like shapes avoided (exact match)
    real = torch.tensor([[1.0],
                         [2.0],
                         [3.0]], dtype=torch.float64).numpy()
    imag = torch.tensor([[0.0],
                         [-2.0],
                         [4.0]], dtype=torch.float64).numpy()
    out = torch.empty(real.shape, dtype=torch.complex128).numpy()
    list_of_inputs.append(copy.deepcopy({"real": real, "imag": imag, "out": out}))

    # Input 7: 4D float32
    real = torch.arange(12, dtype=torch.float32).view(2, 1, 3, 2).numpy()
    imag = (-torch.arange(12, dtype=torch.float32).view(2, 1, 3, 2)).numpy()
    out = torch.empty(real.shape, dtype=torch.complex64).numpy()
    list_of_inputs.append(copy.deepcopy({"real": real, "imag": imag, "out": out}))

    # Input 8: non-contiguous float32 via slicing
    base_r = torch.arange(12, dtype=torch.float32).view(3, 4)
    base_i = torch.arange(100, 112, dtype=torch.float32).view(3, 4)
    real = base_r[:, ::2].numpy()
    imag = base_i[:, ::2].numpy()
    out = torch.empty(real.shape, dtype=torch.complex64).numpy()
    list_of_inputs.append(copy.deepcopy({"real": real, "imag": imag, "out": out}))

    # Input 9: empty float32 (0, 3)
    real = torch.empty((0, 3), dtype=torch.float32).numpy()
    imag = torch.empty((0, 3), dtype=torch.float32).numpy()
    out = torch.empty((0, 3), dtype=torch.complex64).numpy()
    list_of_inputs.append(copy.deepcopy({"real": real, "imag": imag, "out": out}))

    # Input 10: contains NaN and Inf float64
    real = torch.tensor([float('nan'), float('inf'), float('-inf')], dtype=torch.float64).numpy()
    imag = torch.tensor([0.0, -1.0, 2.5], dtype=torch.float64).numpy()
    out = torch.empty(real.shape, dtype=torch.complex128).numpy()
    list_of_inputs.append(copy.deepcopy({"real": real, "imag": imag, "out": out}))

    # Input 11: transpose (non-contiguous) float32
    real = torch.arange(12, dtype=torch.float32).view(3, 4).t().numpy()
    imag = (-torch.arange(12, dtype=torch.float32).view(3, 4).t()).numpy()
    out = torch.empty(real.shape, dtype=torch.complex64).numpy()
    list_of_inputs.append(copy.deepcopy({"real": real, "imag": imag, "out": out}))

    # Input 12: scalar float64
    real = torch.tensor(-0.0, dtype=torch.float64).numpy()
    imag = torch.tensor(3.141592653589793, dtype=torch.float64).numpy()
    out = torch.empty((), dtype=torch.complex128).numpy()
    list_of_inputs.append(copy.deepcopy({"real": real, "imag": imag, "out": out}))

    return list_of_inputs

generated_inputs["torch.complex"] = complex_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.complex' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.complex'.")


check_valid('torch.complex', generated_inputs['torch.complex'], lib="torch", suffix=0)
