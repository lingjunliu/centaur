
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def masked_select_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor and boolean mask
    input_tensor = torch.randn(3, 4).numpy()
    mask = (torch.randn(3, 4) > 0).numpy()
    input_dict = {"input": input_tensor, "mask": mask}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Integer tensor and boolean mask
    input_tensor = torch.randint(-5, 5, (2, 2)).numpy()
    mask = (torch.randint(0, 2, (2, 2)) == 1).numpy()
    input_dict = {"input": input_tensor, "mask": mask}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D tensor
    input_tensor = torch.arange(10).float().numpy()
    mask = (torch.arange(10) % 2 == 0).numpy()
    input_dict = {"input": input_tensor, "mask": mask}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D tensor and mask
    input_tensor = torch.randn(2, 3, 4).numpy()
    mask = (torch.randn(2, 3, 4) > 0.5).numpy()
    input_dict = {"input": input_tensor, "mask": mask}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Tensor with negative values and a mask selecting negative values
    input_tensor = torch.randn(5, 5).numpy()
    mask = (input_tensor < 0)
    input_dict = {"input": input_tensor, "mask": mask}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different shape tensors. Mask can be broadcastable.
    input_tensor = torch.randn(4, 4).numpy()
    mask = torch.tensor([True, False, True, False]).numpy()
    input_dict = {"input": input_tensor, "mask": mask}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.masked_select"] = masked_select_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.masked_select' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.masked_select'.")

check_valid('torch.masked_select', generated_inputs['torch.masked_select'], lib="torch")
