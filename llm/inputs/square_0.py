
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def square_inputs():
    list_of_inputs = []

    # Input 1: 1D tensor, positive values, no out
    input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0]).numpy()
    out_tensor = torch.tensor([]).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D tensor, negative values, no out
    input_tensor = torch.tensor([-1.0, -2.0, -3.0, -4.0]).numpy()
    out_tensor = torch.tensor([]).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D tensor, mixed values, no out
    input_tensor = torch.tensor([[-1.0, 2.0], [3.0, -4.0]]).numpy()
    out_tensor = torch.tensor([]).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D tensor, positive values, pre-allocated out
    input_tensor = torch.tensor([1.0, 2.0, 3.0]).numpy()
    out_tensor = torch.zeros(3).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D tensor, negative values, pre-allocated out
    input_tensor = torch.tensor([[-1.0, -2.0], [-3.0, -4.0]]).numpy()
    out_tensor = torch.zeros((2, 2)).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D tensor, mixed values, no out
    input_tensor = torch.randn(2, 3, 4).numpy()
    out_tensor = torch.tensor([]).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D tensor with zeros
    input_tensor = torch.tensor([0.0, 0.0, 0.0]).numpy()
    out_tensor = torch.tensor([]).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 0D tensor (scalar), positive
    input_tensor = torch.tensor(5.0).numpy()
    out_tensor = torch.tensor([]).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 0D tensor (scalar), negative
    input_tensor = torch.tensor(-5.0).numpy()
    out_tensor = torch.tensor([]).numpy()
    input_dict = {"input": input_tensor, "out": out_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.square"] = square_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.square' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.square'.")

check_valid('torch.square', generated_inputs['torch.square'], lib="torch", suffix=0)
