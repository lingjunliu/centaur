
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def sign_inputs():
    list_of_inputs = []

    # Input 1: 1D tensor with positive, negative, and zero values
    input_tensor = torch.tensor([0.7, -1.2, 0.0, 2.3], dtype=torch.float32).numpy()
    out_tensor = np.array([], dtype=np.float32)
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D tensor with mixed values
    input_tensor = torch.tensor([[-1.0, 2.0], [0.0, -3.0]], dtype=torch.float32).numpy()
    out_tensor = np.array([], dtype=np.float32)
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D tensor with all positive values
    input_tensor = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=torch.float32).numpy()
    out_tensor = np.array([], dtype=np.float32)
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 1D tensor with only zeros
    input_tensor = torch.zeros(5, dtype=torch.float32).numpy()
    out_tensor = np.array([], dtype=np.float32)
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 1D tensor with only negative values
    input_tensor = torch.tensor([-1.0, -2.0, -3.0], dtype=torch.float32).numpy()
    out_tensor = np.array([], dtype=np.float32)
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.sign"] = sign_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.sign' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sign'.")

check_valid('torch.sign', generated_inputs['torch.sign'], lib="torch", suffix=0)
