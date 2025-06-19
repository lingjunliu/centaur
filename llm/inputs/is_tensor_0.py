
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def is_tensor_inputs():
    list_of_inputs = []

    # Input 1: Float tensor
    input_tensor = torch.randn(3, 4).numpy()
    input_dict = {"obj": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Int tensor
    input_tensor = torch.randint(0, 10, (2, 2)).numpy()
    input_dict = {"obj": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Complex tensor
    input_tensor = torch.randn(5, 5, dtype=torch.complex64).numpy()
    input_dict = {"obj": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Bool tensor
    input_tensor = torch.tensor([[True, False], [False, True]]).numpy()
    input_dict = {"obj": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D tensor
    input_tensor = torch.arange(5).numpy()
    input_dict = {"obj": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D tensor
    input_tensor = torch.randn(2, 3, 4).numpy()
    input_dict = {"obj": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Zero dim tensor
    input_tensor = torch.tensor(5).numpy()
    input_dict = {"obj": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.is_tensor"] = is_tensor_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.is_tensor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.is_tensor'.")

check_valid('torch.is_tensor', generated_inputs['torch.is_tensor'], lib="torch")
