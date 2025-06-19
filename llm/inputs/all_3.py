
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_all_inputs():
    list_of_inputs = []

    # Case 1: Basic boolean tensor, no dim
    input_tensor = torch.tensor([True, True, False, True]).bool().numpy()
    input_dict = {"input": input_tensor, "dim": None, "keepdim": False, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: 2D boolean tensor, dim=0
    input_tensor = torch.tensor([[True, True], [False, True], [True, True]]).bool().numpy()
    input_dict = {"input": input_tensor, "dim": (0,), "keepdim": False, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: 2D boolean tensor, dim=1, keepdim=True
    input_tensor = torch.tensor([[True, True], [False, True], [True, True]]).bool().numpy()
    input_dict = {"input": input_tensor, "dim": (1,), "keepdim": True, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: 3D boolean tensor, dim=(0, 2)
    input_tensor = torch.tensor([[[True, True], [False, True]], [[True, True], [True, False]]]).bool().numpy()
    input_dict = {"input": input_tensor, "dim": (0, 2), "keepdim": False, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Integer tensor (will be converted to boolean), no dim
    input_tensor = torch.tensor([1, 1, 0, 1]).int().numpy()
    input_dict = {"input": input_tensor, "dim": None, "keepdim": False, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: Float tensor (will be converted to boolean), dim=0
    input_tensor = torch.tensor([1.0, 0.0, 1.0]).float().numpy()
    input_dict = {"input": input_tensor, "dim": (0,), "keepdim": False, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 7: uint8 tensor, no dim
    input_tensor = torch.tensor([1, 1, 0, 1]).byte().numpy()
    input_dict = {"input": input_tensor, "dim": None, "keepdim": False, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 8: Empty tensor
    input_tensor = torch.tensor([]).bool().numpy()
    input_dict = {"input": input_tensor, "dim": None, "keepdim": False, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.all_3"] = torch_all_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.all_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.all_3'.")

check_valid('torch.all', generated_inputs['torch.all_3'], lib="torch")
