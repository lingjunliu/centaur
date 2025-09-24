
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def deg2rad_inputs():
    list_of_inputs = []

    # Input 1: 1D tensor
    input_tensor = torch.tensor([0.0, 30.0, 45.0, 60.0, 90.0]).numpy()
    out_tensor = torch.tensor([]).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D tensor
    input_tensor = torch.tensor([[180.0, -180.0], [360.0, -360.0]]).numpy()
    out_tensor = torch.tensor([[0.0, 0.0], [0.0, 0.0]]).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D tensor
    input_tensor = torch.tensor([[[0.0, 90.0], [180.0, 270.0]], [[360.0, -90.0], [-180.0, -270.0]]]).numpy()
    out_tensor = torch.tensor([[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Tensor with large values
    input_tensor = torch.tensor([720.0, 1080.0, 1440.0]).numpy()
    out_tensor = torch.tensor([]).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Tensor with negative and positive values, including 0.
    input_tensor = torch.tensor([-45.0, 0.0, 45.0, -90.0, 90.0]).numpy()
    out_tensor = torch.tensor([]).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Empty tensor
    input_tensor = torch.tensor([]).numpy()
    out_tensor = torch.tensor([]).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Tensor with float64
    input_tensor = torch.tensor([180.0, 360.0], dtype=torch.float64).numpy()
    out_tensor = torch.tensor([0.0, 0.0], dtype=torch.float64).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.deg2rad"] = deg2rad_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.deg2rad' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.deg2rad'.")

check_valid('torch.deg2rad', generated_inputs['torch.deg2rad'], lib="torch", suffix=0)
