
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def is_storage_inputs():
    list_of_inputs = []

    # Input 1: Float tensor
    input1 = torch.randn(3, 4).numpy()
    input_dict1 = {"obj": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Int tensor
    input2 = torch.randint(0, 10, (2, 2)).numpy()
    input_dict2 = {"obj": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Complex tensor
    input3 = torch.randn(5, 5, dtype=torch.complex64).numpy()
    input_dict3 = {"obj": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 1D tensor
    input4 = torch.randn(10).numpy()
    input_dict4 = {"obj": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 3D tensor
    input5 = torch.randn(2, 3, 4).numpy()
    input_dict5 = {"obj": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Empty tensor
    input6 = torch.empty(0).numpy()
    input_dict6 = {"obj": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: Bool tensor
    input7 = torch.tensor([True, False, True]).numpy()
    input_dict7 = {"obj": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs["torch.is_storage"] = is_storage_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.is_storage' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.is_storage'.")

check_valid('torch.is_storage', generated_inputs['torch.is_storage'], lib="torch")
