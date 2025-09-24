
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def movedim_inputs():
    list_of_inputs = []

    # Input 1: Basic case with a 3D tensor
    input1 = torch.randn(3, 2, 1).numpy()
    source1 = (1,)
    destination1 = (0,)
    input_dict1 = {"input": input1, "source": source1, "destination": destination1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Moving multiple dimensions
    input2 = torch.randn(3, 2, 4, 5).numpy()
    source2 = (1, 3)
    destination2 = (0, 1)
    input_dict2 = {"input": input2, "source": source2, "destination": destination2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Using negative indices
    input3 = torch.randn(3, 2, 4).numpy()
    source3 = (-1,)
    destination3 = (0,)
    input_dict3 = {"input": input3, "source": source3, "destination": destination3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Moving to the same position (no change)
    input4 = torch.randn(2, 3, 4).numpy()
    source4 = (0,)
    destination4 = (0,)
    input_dict4 = {"input": input4, "source": source4, "destination": destination4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Higher dimensional tensor
    input5 = torch.randn(2, 3, 4, 5, 6).numpy()
    source5 = (2, 4)
    destination5 = (0, 1)
    input_dict5 = {"input": input5, "source": source5, "destination": destination5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.movedim_2"] = movedim_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.movedim_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.movedim_2'.")

check_valid('torch.movedim', generated_inputs['torch.movedim_2'], lib="torch")
