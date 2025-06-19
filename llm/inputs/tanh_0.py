
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def tanh_inputs():
    list_of_inputs = []

    # Input 1: 1D tensor, out=None
    input_tensor = torch.randn(4).numpy()
    out_tensor = np.array([])

    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D tensor, out with same shape
    input_tensor = torch.randn(2, 3).numpy()
    out_tensor = torch.randn(2, 3).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D tensor, out is pre-allocated
    input_tensor = torch.randn(2, 3, 4).numpy()
    out_tensor = torch.zeros(2, 3, 4).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 1D tensor with negative values, out is empty
    input_tensor = torch.tensor([-1.0, -2.0, -3.0, -4.0]).numpy()
    out_tensor = np.array([])
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Large tensor
    input_tensor = torch.randn(100, 100).numpy()
    out_tensor = np.array([])
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.tanh"] = tanh_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.tanh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.tanh'.")

check_valid('torch.tanh', generated_inputs['torch.tanh'], lib="torch", suffix=0)
