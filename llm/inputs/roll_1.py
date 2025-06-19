
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_roll_inputs():
    list_of_inputs = []

    # Case 1: 1D tensor, positive shift
    input_tensor = torch.tensor([1, 2, 3, 4, 5]).numpy()
    shifts = 2
    dims = 0
    input_dict = {"input": input_tensor, "shifts": shifts, "dims": dims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: 2D tensor, negative shift
    input_tensor = torch.tensor([[1, 2], [3, 4]]).numpy()
    shifts = -1
    dims = 0
    input_dict = {"input": input_tensor, "shifts": shifts, "dims": dims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: 3D tensor, multiple shifts and dims
    input_tensor = torch.randn(2, 3, 4).numpy()
    shifts = (1, -1)
    dims = (0, 1)
    input_dict = {"input": input_tensor, "shifts": shifts, "dims": dims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: 2D tensor, shift along columns
    input_tensor = torch.arange(12).reshape(3, 4).numpy()
    shifts = 1
    dims = 1
    input_dict = {"input": input_tensor, "shifts": shifts, "dims": dims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: 1D tensor, large shift
    input_tensor = torch.tensor([1, 2, 3, 4, 5]).numpy()
    shifts = 7
    dims = 0
    input_dict = {"input": input_tensor, "shifts": shifts, "dims": dims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: 2D tensor, tuple shifts
    input_tensor = torch.arange(16).reshape(4, 4).numpy()
    shifts = (1, 2)
    dims = (0, 1)
    input_dict = {"input": input_tensor, "shifts": shifts, "dims": dims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7: 3D tensor, single shift
    input_tensor = torch.randn(2, 2, 2).numpy()
    shifts = 1
    dims = 2
    input_dict = {"input": input_tensor, "shifts": shifts, "dims": dims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.roll_1"] = torch_roll_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.roll_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.roll_1'.")

check_valid('torch.roll', generated_inputs['torch.roll_1'], lib="torch")
