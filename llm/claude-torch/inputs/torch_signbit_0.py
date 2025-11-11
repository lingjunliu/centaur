
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import torch
import numpy as np
import copy

def signbit_inputs():
    list_of_inputs = []
    
    input_tensor = torch.tensor([0.7, -1.2, 0., 2.3]).numpy()
    out_tensor = torch.tensor([False, False, False, False]).numpy()
    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.tensor([-0.0, 0.0]).numpy()
    out_tensor = torch.tensor([False, False]).numpy()
    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]]).numpy()
    out_tensor = torch.tensor([[False, False, False], [False, False, False]]).numpy()
    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.tensor([-1.0, -2.0, -3.0, -4.0]).numpy()
    out_tensor = torch.tensor([False, False, False, False]).numpy()
    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0]).numpy()
    out_tensor = torch.tensor([False, False, False, False]).numpy()
    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.tensor([[[1.0, -2.0], [3.0, -4.0]], [[5.0, -6.0], [7.0, -8.0]]]).numpy()
    out_tensor = torch.tensor([[[False, False], [False, False]], [[False, False], [False, False]]]).numpy()
    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.tensor([-5.0]).numpy()
    out_tensor = torch.tensor([False]).numpy()
    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.tensor([1e10, -1e10, 1e-10, -1e-10]).numpy()
    out_tensor = torch.tensor([False, False, False, False]).numpy()
    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.tensor([1, -2, 3, -4, 0]).numpy()
    out_tensor = torch.tensor([False, False, False, False, False]).numpy()
    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.tensor([0.5, -0.5, 1.5, -1.5]).numpy()
    input_dict = {
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.signbit' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.signbit'.")


check_valid('torch.signbit', generated_inputs['torch.signbit'], lib="torch", suffix=0)
