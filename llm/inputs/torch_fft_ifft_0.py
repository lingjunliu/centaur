
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def ifft_inputs():
    list_of_inputs = []

    # Input 1
    input = torch.tensor([6.+0.j, -2.+2.j, -2.+0.j, -2.-2.j]).numpy()
    n = None
    dim = -1
    norm = None
    out = torch.tensor([0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]).numpy()

    input_dict = {
        "input": input,
        "n": n,
        "dim": dim,
        "norm": norm,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input = torch.randn(8, dtype=torch.complex64).numpy()
    n = 8
    dim = 0
    norm = "backward"
    out = torch.randn(8, dtype=torch.complex64).numpy()
    input_dict = {
        "input": input,
        "n": n,
        "dim": dim,
        "norm": norm,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input = torch.randn(4, 4, dtype=torch.complex128).numpy()
    n = 2
    dim = 1
    norm = "forward"
    out = torch.randn(4, 2, dtype=torch.complex128).numpy()
    input_dict = {
        "input": input,
        "n": n,
        "dim": dim,
        "norm": norm,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input = torch.randn(2, 2, 2, dtype=torch.complex64).numpy()
    n = 4
    dim = 2
    norm = "ortho"
    out = torch.randn(2, 2, 4, dtype=torch.complex64).numpy()
    input_dict = {
        "input": input,
        "n": n,
        "dim": dim,
        "norm": norm,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input = torch.randn(16, dtype=torch.complex64).numpy()
    n = 4
    dim = 0
    norm = "backward"
    out = torch.randn(4, dtype=torch.complex64).numpy()
    input_dict = {
        "input": input,
        "n": n,
        "dim": dim,
        "norm": norm,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input = torch.randn(1, 8, dtype=torch.complex128).numpy()
    n = None
    dim = 1
    norm = "forward"
    out = torch.randn(1, 8, dtype=torch.complex128).numpy()
    input_dict = {
        "input": input,
        "n": n,
        "dim": dim,
        "norm": norm,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    input = torch.randn(2, 4, dtype=torch.complex64).numpy()
    n = 8
    dim = -1
    norm = "ortho"
    out = torch.randn(2, 8, dtype=torch.complex64).numpy()
    input_dict = {
        "input": input,
        "n": n,
        "dim": dim,
        "norm": norm,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input = torch.randn(32, dtype=torch.complex64).numpy()
    n = 16
    dim = 0
    norm = None
    out = torch.randn(16, dtype=torch.complex64).numpy()
    input_dict = {
        "input": input,
        "n": n,
        "dim": dim,
        "norm": norm,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    input = torch.randn(1, 1, 4, dtype=torch.complex128).numpy()
    n = None
    dim = 2
    norm = "backward"
    out = torch.randn(1, 1, 4, dtype=torch.complex128).numpy()
    input_dict = {
        "input": input,
        "n": n,
        "dim": dim,
        "norm": norm,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input = torch.randn(2, 2, dtype=torch.complex64).numpy()
    n = 4
    dim = 1
    norm = "forward"
    out = torch.randn(2, 4, dtype=torch.complex64).numpy()
    input_dict = {
        "input": input,
        "n": n,
        "dim": dim,
        "norm": norm,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11
    input = torch.randn(2, 2, dtype=torch.complex64).numpy()
    n = 1
    dim = 1
    norm = "forward"
    out = torch.randn(2, 1, dtype=torch.complex64).numpy()
    input_dict = {
        "input": input,
        "n": n,
        "dim": dim,
        "norm": norm,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.fft.ifft"] = ifft_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.fft.ifft' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.fft.ifft'.")

check_valid('torch.fft.ifft', generated_inputs['torch.fft.ifft'], lib="torch", suffix=0)
