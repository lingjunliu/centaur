
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def movedim_inputs():
    list_of_inputs = []

    # Example 1: Basic 3D tensor
    input_tensor = torch.randn(3, 2, 1).numpy()
    source = 1
    destination = 0
    input_dict = {"input": input_tensor, "source": source, "destination": destination}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2: Move multiple dimensions
    input_tensor = torch.randn(3, 2, 4, 5).numpy()
    source = (1, 2)
    destination = (0, 1)
    input_dict = {"input": input_tensor, "source": source, "destination": destination}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 3: Move to the end
    input_tensor = torch.randn(3, 2, 4).numpy()
    source = 0
    destination = 2
    input_dict = {"input": input_tensor, "source": source, "destination": destination}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 4: Using negative indices
    input_tensor = torch.randn(3, 2, 4).numpy()
    source = -1
    destination = 0
    input_dict = {"input": input_tensor, "source": source, "destination": destination}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 5: Complex tensor
    input_tensor = torch.randn(2, 3, dtype=torch.complex64).numpy()
    source = 1
    destination = 0
    input_dict = {"input": input_tensor, "source": source, "destination": destination}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Example 6: Integer tensor
    input_tensor = torch.randint(0, 10, (4, 5, 2)).numpy()
    source = 0
    destination = 2
    input_dict = {"input": input_tensor, "source": source, "destination": destination}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 7: Higher dimensions
    input_tensor = torch.randn(2, 3, 4, 5, 6).numpy()
    source = (0, 2, 4)
    destination = (1, 3, 0)
    input_dict = {"input": input_tensor, "source": source, "destination": destination}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.movedim_1"] = movedim_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.movedim_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.movedim_1'.")

check_valid('torch.movedim', generated_inputs['torch.movedim_1'], lib="torch")
