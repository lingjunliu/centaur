
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def rad2deg_inputs():
    list_of_inputs = []

    # Input 1: 1D tensor
    input_tensor = torch.tensor([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    out_tensor = torch.zeros_like(input_tensor).numpy()
    input_dict = {"input": input_tensor.numpy(), "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D tensor with negative values
    input_tensor = torch.tensor([[-np.pi, -np.pi/2], [0, np.pi/4]])
    out_tensor = torch.zeros_like(input_tensor).numpy()
    input_dict = {"input": input_tensor.numpy(), "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D tensor
    input_tensor = torch.tensor([[[0, np.pi/2], [np.pi, 3*np.pi/2]], [[2*np.pi, -np.pi], [-np.pi/2, 0]]])
    out_tensor = torch.zeros_like(input_tensor).numpy()
    input_dict = {"input": input_tensor.numpy(), "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Empty tensor
    input_tensor = torch.tensor([])
    out_tensor = torch.tensor([]).numpy()
    input_dict = {"input": input_tensor.numpy(), "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Tensor with some large values
    input_tensor = torch.tensor([100*np.pi, -50*np.pi, 25.5*np.pi])
    out_tensor = torch.zeros_like(input_tensor).numpy()
    input_dict = {"input": input_tensor.numpy(), "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.rad2deg"] = rad2deg_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.rad2deg' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.rad2deg'.")

check_valid('torch.rad2deg', generated_inputs['torch.rad2deg'], lib="torch", suffix=0)
