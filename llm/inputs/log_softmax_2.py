
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def log_softmax_inputs():
    list_of_inputs = []

    # Input 1: 1D tensor
    input_tensor = torch.randn(5).numpy().astype(np.float32)
    dim = 0
    input_dict = {"input": input_tensor, "dim": int(dim)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D tensor
    input_tensor = torch.randn(3, 4).numpy().astype(np.float32)
    dim = 1
    input_dict = {"input": input_tensor, "dim": int(dim)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D tensor
    input_tensor = torch.randn(2, 3, 4).numpy().astype(np.float32)
    dim = 0
    input_dict = {"input": input_tensor, "dim": int(dim)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 4D tensor
    input_tensor = torch.randn(2, 3, 4, 5).numpy().astype(np.float32)
    dim = 2
    input_dict = {"input": input_tensor, "dim": int(dim)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative values
    input_tensor = torch.randn(2, 2) * -1.0
    input_tensor = input_tensor.numpy().astype(np.float32)
    dim = 1
    input_dict = {"input": input_tensor, "dim": int(dim)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.functional.log_softmax_2"] = log_softmax_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.log_softmax_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.log_softmax_2'.")

check_valid('torch.nn.functional.log_softmax', generated_inputs['torch.nn.functional.log_softmax_2'], lib="torch", suffix=2)
