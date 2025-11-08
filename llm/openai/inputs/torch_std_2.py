
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_std_inputs():
    list_of_inputs = []

    # Input 1
    input_arr = torch.tensor([1.0, -2.0, 3.5, 4.0, -5.5], dtype=torch.float32).numpy()
    dim = (0,)
    correction = 1
    keepdim = False
    out = torch.tensor(0.0, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "input": input_arr,
        "dim": dim,
        "correction": correction,
        "keepdim": keepdim,
        "out": out
    }))

    # Input 2
    input_arr = torch.arange(12, dtype=torch.float32).reshape(3, 4).numpy()
    dim = (1,)
    correction = 0
    keepdim = True
    out = torch.zeros((3, 1), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "input": input_arr,
        "dim": dim,
        "correction": correction,
        "keepdim": keepdim,
        "out": out
    }))

    # Input 3
    input_arr = torch.linspace(-5, 5, steps=12, dtype=torch.float64).reshape(3, 4).numpy()
    dim = (-1,)
    correction = 1
    keepdim = False
    out = torch.zeros((3,), dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({
        "input": input_arr,
        "dim": dim,
        "correction": correction,
        "keepdim": keepdim,
        "out": out
    }))

    # Input 4
    input_arr = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).numpy()
    dim = (0, 2)
    correction = 1
    keepdim = False
    out = torch.zeros((3,), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "input": input_arr,
        "dim": dim,
        "correction": correction,
        "keepdim": keepdim,
        "out": out
    }))

    # Input 5
    input_arr = torch.randn(2, 3, 4, dtype=torch.float64).numpy()
    dim = (1,)
    correction = 0
    keepdim = True
    out = torch.zeros((2, 1, 4), dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({
        "input": input_arr,
        "dim": dim,
        "correction": correction,
        "keepdim": keepdim,
        "out": out
    }))

    # Input 6
    input_arr = torch.arange(2 * 2 * 3 * 4, dtype=torch.float32).reshape(2, 2, 3, 4).numpy()
    dim = (-1, -2)
    correction = 1
    keepdim = False
    out = torch.zeros((2, 2), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "input": input_arr,
        "dim": dim,
        "correction": correction,
        "keepdim": keepdim,
        "out": out
    }))

    # Input 7
    input_arr = torch.randn(5, 4, 3, dtype=torch.float32).numpy()
    dim = (0, 1)
    correction = 2
    keepdim = False
    out = torch.zeros((3,), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "input": input_arr,
        "dim": dim,
        "correction": correction,
        "keepdim": keepdim,
        "out": out
    }))

    # Input 8
    input_arr = torch.tensor([[-3.0], [0.0], [3.0], [6.0]], dtype=torch.float64).numpy()
    dim = (-2,)
    correction = 1
    keepdim = True
    out = torch.zeros((1, 1), dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({
        "input": input_arr,
        "dim": dim,
        "correction": correction,
        "keepdim": keepdim,
        "out": out
    }))

    # Input 9
    input_arr = torch.randn(2, 1, 3, 1, 4, dtype=torch.float32).numpy()
    dim = (1, 3)
    correction = 0
    keepdim = False
    out = torch.zeros((2, 3, 4), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "input": input_arr,
        "dim": dim,
        "correction": correction,
        "keepdim": keepdim,
        "out": out
    }))

    # Input 10
    input_arr = torch.tensor([-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, -15.0, 20.0], dtype=torch.float32).numpy()
    dim = (0,)
    correction = 0
    keepdim = True
    out = torch.zeros((1,), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "input": input_arr,
        "dim": dim,
        "correction": correction,
        "keepdim": keepdim,
        "out": out
    }))

    # Input 11
    input_arr = torch.tensor([[1.5, -2.5], [3.0, 4.5]], dtype=torch.float64).numpy()
    dim = (0, 1)
    correction = 0
    keepdim = False
    out = torch.tensor(0.0, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({
        "input": input_arr,
        "dim": dim,
        "correction": correction,
        "keepdim": keepdim,
        "out": out
    }))

    # Input 12
    input_arr = torch.empty((4, 0, 5), dtype=torch.float32).numpy()
    dim = (2,)
    correction = 0
    keepdim = False
    out = torch.zeros((4, 0), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "input": input_arr,
        "dim": dim,
        "correction": correction,
        "keepdim": keepdim,
        "out": out
    }))

    return list_of_inputs

generated_inputs["torch.std_2"] = torch_std_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.std_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.std_2'.")


check_valid('torch.std', generated_inputs['torch.std_2'], lib="torch", suffix=2)
