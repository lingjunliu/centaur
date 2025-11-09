
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import torch
import numpy as np
import copy


def polygamma_inputs():
    list_of_inputs = []
    
    n = 0
    input_tensor = torch.tensor([1.0, 2.0, 3.0]).numpy()
    out = torch.empty(3).numpy()
    input_dict = {
        "n": n,
        "input": input_tensor,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    n = 1
    input_tensor = torch.tensor([[1.5, 2.5], [3.5, 4.5]]).numpy()
    out = torch.empty(2, 2).numpy()
    input_dict = {
        "n": n,
        "input": input_tensor,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    n = 2
    input_tensor = torch.tensor([0.5, 1.0, 1.5, 2.0]).numpy()
    out = torch.empty(4).numpy()
    input_dict = {
        "n": n,
        "input": input_tensor,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    n = 3
    input_tensor = torch.tensor([5.0, 10.0, 15.0]).numpy()
    out = torch.empty(3).numpy()
    input_dict = {
        "n": n,
        "input": input_tensor,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    n = 0
    input_tensor = torch.ones((2, 3, 4)).numpy()
    out = torch.empty(2, 3, 4).numpy()
    input_dict = {
        "n": n,
        "input": input_tensor,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    n = 1
    input_tensor = torch.tensor([2.0, 4.0, 6.0, 8.0]).numpy()
    input_dict = {
        "n": n,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    n = 4
    input_tensor = torch.tensor(3.0).numpy()
    out = torch.empty(()).numpy()
    input_dict = {
        "n": n,
        "input": input_tensor,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    n = 0
    input_tensor = torch.linspace(1, 10, 20).reshape(4, 5).numpy()
    out = torch.empty(4, 5).numpy()
    input_dict = {
        "n": n,
        "input": input_tensor,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    n = 2
    input_tensor = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]).numpy()
    out = torch.empty(5).numpy()
    input_dict = {
        "n": n,
        "input": input_tensor,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    n = 5
    input_tensor = torch.tensor([1.5, 2.5, 3.5]).numpy()
    input_dict = {
        "n": n,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs


generated_inputs = {}
generated_inputs["torch.special.polygamma"] = polyg

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.special.polygamma' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.polygamma'.")


check_valid('torch.special.polygamma', generated_inputs['torch.special.polygamma'], lib="torch", suffix=0)
