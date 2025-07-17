
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def irfftn_inputs():
    list_of_inputs = []

    # Input 1
    input = torch.randn(5, 5, 3, dtype=torch.complex64).numpy()
    s = (5, 5, 5)
    dim = (0, 1, 2)
    norm = "backward"
    out = torch.empty(s).numpy()

    input_dict = {
        "input": input,
        "s": s,
        "dim": dim,
        "norm": norm,
        "out": out,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input = torch.randn(10, 7, dtype=torch.complex64).numpy()
    s = (10, 12)
    dim = (0, 1)
    norm = "forward"
    out = torch.empty(s).numpy()

    input_dict = {
        "input": input,
        "s": s,
        "dim": dim,
        "norm": norm,
        "out": out,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input = torch.randn(3, 4, 5, dtype=torch.complex64).numpy()
    s = (3, 4, 5)
    dim = (0, 1, 2)
    norm = "ortho"
    out = torch.empty(s).numpy()
    input_dict = {
        "input": input,
        "s": s,
        "dim": dim,
        "norm": norm,
        "out": out,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input = torch.randn(2, 3, dtype=torch.complex64).numpy()
    s = (4, 4)
    dim = (0, 1)
    norm = "backward"
    out = torch.empty(s).numpy()

    input_dict = {
        "input": input,
        "s": s,
        "dim": dim,
        "norm": norm,
        "out": out,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 5
    input = torch.randn(4, 6, dtype=torch.complex64).numpy()
    s = (4, 6)
    dim = (0, 1)
    norm = "forward"
    out = torch.empty(s).numpy()

    input_dict = {
        "input": input,
        "s": s,
        "dim": dim,
        "norm": norm,
        "out": out,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    input = torch.randn(2, 3, 4, dtype=torch.complex64).numpy()
    s = (2, 3, 4)
    dim = (0, 1, 2)
    norm = "ortho"
    out = torch.empty(s).numpy()
    input_dict = {
        "input": input,
        "s": s,
        "dim": dim,
        "norm": norm,
        "out": out,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input = torch.randn(5, 5, dtype=torch.complex64).numpy()
    s = (5, 5)
    dim = (0, 1)
    norm = "backward"
    out = torch.empty(s).numpy()

    input_dict = {
        "input": input,
        "s": s,
        "dim": dim,
        "norm": norm,
        "out": out,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input = torch.randn(8, dtype=torch.complex64).numpy()
    s = (8,)
    dim = (0,)
    norm = "forward"
    out = torch.empty(s).numpy()

    input_dict = {
        "input": input,
        "s": s,
        "dim": dim,
        "norm": norm,
        "out": out,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input = torch.randn(1, 2, 3, dtype=torch.complex64).numpy()
    s = (1, 2, 3)
    dim = (0, 1, 2)
    norm = "ortho"
    out = torch.empty(s).numpy()
    input_dict = {
        "input": input,
        "s": s,
        "dim": dim,
        "norm": norm,
        "out": out,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    input = torch.randn(2, 2, dtype=torch.complex64).numpy()
    s = (2, 2)
    dim = (0, 1)
    norm = "backward"
    out = torch.empty(s).numpy()

    input_dict = {
        "input": input,
        "s": s,
        "dim": dim,
        "norm": norm,
        "out": out,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.fft.irfftn"] = irfftn_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.fft.irfftn' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.fft.irfftn'.")

check_valid('torch.fft.irfftn', generated_inputs['torch.fft.irfftn'], lib="torch", suffix=0)
