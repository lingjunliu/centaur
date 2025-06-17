
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_amin_inputs():
    list_of_inputs = []

    # Test case 1: 2D float tensor, dim=0, keepdim=False
    input_tensor = torch.randn(4, 4).numpy()
    dim = 0
    keepdim = False
    input_dict = {"input": input_tensor, "dim": dim, "keepdim": keepdim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 2: 2D int tensor, dim=1, keepdim=True
    input_tensor = torch.randint(-5, 5, (3, 5)).numpy()
    dim = 1
    keepdim = True
    input_dict = {"input": input_tensor, "dim": dim, "keepdim": keepdim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 3: 3D float tensor, dim=(0, 2), keepdim=False
    input_tensor = torch.randn(2, 3, 4).numpy()
    dim = (0, 2)
    keepdim = False
    input_dict = {"input": input_tensor, "dim": dim, "keepdim": keepdim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 4: 4D int tensor, dim=(1, 3), keepdim=True
    input_tensor = torch.randint(-10, 10, (2, 3, 2, 4)).numpy()
    dim = (1, 3)
    keepdim = True
    input_dict = {"input": input_tensor, "dim": dim, "keepdim": keepdim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 5: 1D float tensor, dim=0, keepdim=False
    input_tensor = torch.randn(5).numpy()
    dim = 0
    keepdim = False
    input_dict = {"input": input_tensor, "dim": dim, "keepdim": keepdim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.amin_1"] = torch_amin_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.amin_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.amin_1'.")

check_valid('torch.amin', generated_inputs['torch.amin_1'], lib="torch")
