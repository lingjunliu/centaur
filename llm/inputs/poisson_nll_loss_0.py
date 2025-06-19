
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def poisson_nll_loss_inputs():
    list_of_inputs = []

    # Case 1: Basic case with log_input=True
    input_dict = {
        "input": np.random.randn(2, 3).astype(np.float32),
        "target": np.random.randint(0, 5, size=(2, 3)).astype(np.float32),
        "log_input": True,
        "full": False,
        "eps": 1e-8,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Basic case with log_input=False
    input_dict = {
        "input": np.random.rand(2, 3).astype(np.float32),
        "target": np.random.randint(0, 5, size=(2, 3)).astype(np.float32),
        "log_input": False,
        "full": False,
        "eps": 1e-8,
        "reduction": 'sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Different reduction, 1D input
    input_dict = {
        "input": np.random.randn(5).astype(np.float32),
        "target": np.random.randint(0, 5, size=(5)).astype(np.float32),
        "log_input": True,
        "full": True,
        "eps": 1e-6,
        "reduction": 'none'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Different shape, different eps
    input_dict = {
        "input": np.random.rand(1, 1, 5, 5).astype(np.float32),
        "target": np.random.randint(0, 5, size=(1, 1, 5, 5)).astype(np.float32),
        "log_input": False,
        "full": True,
        "eps": 1e-4,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 5: Check for zero target and different reduction mode
    input_dict = {
        "input": np.random.randn(2, 3).astype(np.float32),
        "target": np.zeros((2, 3)).astype(np.float32),
        "log_input": True,
        "full": False,
        "eps": 1e-8,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.functional.poisson_nll_loss"] = poisson_nll_loss_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.poisson_nll_loss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.poisson_nll_loss'.")

check_valid('torch.nn.functional.poisson_nll_loss', generated_inputs['torch.nn.functional.poisson_nll_loss'], lib="torch")
