
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import torch
import copy
import numpy as np

def kl_div_inputs():
    list_of_inputs = []
    
    input = torch.tensor([0.5, 0.3, 0.2]).log().numpy()
    target = torch.tensor([0.6, 0.2, 0.2]).numpy()
    input_dict = {
        "input": input,
        "target": target,
        "size_average": True,
        "reduce": True,
        "reduction": 'mean',
        "log_target": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.randn(3, 4).numpy()
    target = torch.randn(3, 4).numpy()
    input_dict = {
        "input": input,
        "target": target,
        "size_average": False,
        "reduce": True,
        "reduction": 'sum',
        "log_target": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([[0.1, 0.2], [0.3, 0.4]]).log().numpy()
    target = torch.tensor([[0.5, 0.5], [0.6, 0.4]]).numpy()
    input_dict = {
        "input": input,
        "target": target,
        "size_average": True,
        "reduce": False,
        "reduction": 'none',
        "log_target": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([0.2, 0.3, 0.5]).log().numpy()
    target = torch.tensor([0.3, 0.3, 0.4]).log().numpy()
    input_dict = {
        "input": input,
        "target": target,
        "size_average": True,
        "reduce": True,
        "reduction": 'mean',
        "log_target": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.randn(2, 3, 4).numpy()
    target = torch.randn(2, 3, 4).numpy()
    input_dict = {
        "input": input,
        "target": target,
        "size_average": True,
        "reduce": True,
        "reduction": 'batchmean',
        "log_target": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.randn(2, 3, 4, 5).numpy()
    target = torch.randn(2, 3, 4, 5).numpy()
    input_dict = {
        "input": input,
        "target": target,
        "size_average": False,
        "reduce": True,
        "reduction": 'sum',
        "log_target": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.randn(5).numpy()
    target = torch.randn(5).numpy()
    input_dict = {
        "input": input,
        "target": target,
        "size_average": True,
        "reduce": True,
        "reduction": 'mean',
        "log_target": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.randn(10, 5).numpy()
    target = torch.randn(10, 5).numpy()
    input_dict = {
        "input": input,
        "target": target,
        "size_average": False,
        "reduce": False,
        "reduction": 'none',
        "log_target": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.randn(2, 2, 2).numpy()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.kl_div' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.kl_div'.")


check_valid('torch.nn.functional.kl_div', generated_inputs['torch.nn.functional.kl_div'], lib="torch", suffix=0)
