
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def ihfft_inputs():
    list_of_inputs = []

    def create_complex_from_real(arr):
        return arr.astype(np.complex128)

    # Input 1
    input_tensor = create_complex_from_real(np.random.randn(4))
    n_val = None
    dim_val = -1
    norm_val = "backward"
    out_val = torch.tensor([]).numpy()

    input_dict = {
        "input": input_tensor,
        "n": n_val,
        "dim": dim_val,
        "norm": norm_val,
        "out": out_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = create_complex_from_real(np.random.randn(3))
    n_val = 5
    dim_val = 0
    norm_val = "forward"
    out_val = torch.tensor([]).numpy()

    input_dict = {
        "input": input_tensor,
        "n": n_val,
        "dim": dim_val,
        "norm": norm_val,
        "out": out_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = create_complex_from_real(np.random.randn(4))
    n_val = 2
    dim_val = 0
    norm_val = "ortho"
    out_val = torch.tensor([]).numpy()

    input_dict = {
        "input": input_tensor,
        "n": n_val,
        "dim": dim_val,
        "norm": norm_val,
        "out": out_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = create_complex_from_real(np.random.randn(2, 3))
    n_val = None
    dim_val = 1
    norm_val = "backward"
    out_val = torch.tensor([]).numpy()

    input_dict = {
        "input": input_tensor,
        "n": n_val,
        "dim": dim_val,
        "norm": norm_val,
        "out": out_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = create_complex_from_real(np.random.randn(3, 4, 5))
    n_val = 7
    dim_val = 2
    norm_val = "forward"
    out_val = torch.tensor([]).numpy()

    input_dict = {
        "input": input_tensor,
        "n": n_val,
        "dim": dim_val,
        "norm": norm_val,
        "out": out_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = create_complex_from_real(np.random.randn(10))
    n_val = 5
    dim_val = 0
    norm_val = "ortho"
    out_val = torch.tensor([]).numpy()

    input_dict = {
        "input": input_tensor,
        "n": n_val,
        "dim": dim_val,
        "norm": norm_val,
        "out": out_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = create_complex_from_real(np.array([0.0]))
    n_val = None
    dim_val = 0
    norm_val = "backward"
    out_val = torch.tensor([]).numpy()

    input_dict = {
        "input": input_tensor,
        "n": n_val,
        "dim": dim_val,
        "norm": norm_val,
        "out": out_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 8
    input_tensor = create_complex_from_real(np.random.randn(2, 2, 2))
    n_val = None
    dim_val = 1
    norm_val = "forward"
    out_val = torch.tensor([]).numpy()

    input_dict = {
        "input": input_tensor,
        "n": n_val,
        "dim": dim_val,
        "norm": norm_val,
        "out": out_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = create_complex_from_real(np.array([-1.0, -2.0, -3.0, -4.0]))
    n_val = None
    dim_val = -1
    norm_val = "backward"
    out_val = torch.tensor([]).numpy()

    input_dict = {
        "input": input_tensor,
        "n": n_val,
        "dim": dim_val,
        "norm": norm_val,
        "out": out_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = create_complex_from_real(np.random.randn(3))
    n_val = 7
    dim_val = 0
    norm_val = "ortho"
    out_val = torch.tensor([]).numpy()

    input_dict = {
        "input": input_tensor,
        "n": n_val,
        "dim": dim_val,
        "norm": norm_val,
        "out": out_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.fft.ihfft"] = ihfft_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.fft.ihfft' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.fft.ihfft'.")

check_valid('torch.fft.ihfft', generated_inputs['torch.fft.ihfft'], lib="torch", suffix=0)
