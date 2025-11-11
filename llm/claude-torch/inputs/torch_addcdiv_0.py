
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import torch
import copy
import numpy as np

def addcdiv_inputs():
    list_of_inputs = []
    
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    tensor1 = torch.tensor([10.0, 20.0, 30.0]).numpy()
    tensor2 = torch.tensor([2.0, 4.0, 5.0]).numpy()
    value = 1.0
    out = torch.tensor().numpy()
    input_dict = {
        "input": input,
        "tensor1": tensor1,
        "tensor2": tensor2,
        "value": value,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.ones((2, 3)).numpy()
    tensor1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).numpy()
    tensor2 = torch.tensor([[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]]).numpy()
    value = 0.5
    out = torch.tensor().numpy()
    input_dict = {
        "input": input,
        "tensor1": tensor1,
        "tensor2": tensor2,
        "value": value,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.randn(1, 3).numpy()
    tensor1 = torch.randn(3, 1).numpy()
    tensor2 = torch.randn(1, 3).numpy()
    value = 0.1
    out = torch.tensor().numpy()
    input_dict = {
        "input": input,
        "tensor1": tensor1,
        "tensor2": tensor2,
        "value": value,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    tensor1 = torch.tensor([5.0, -10.0, 15.0]).numpy()
    tensor2 = torch.tensor([2.0, 5.0, -3.0]).numpy()
    value = 2.0
    out = torch.tensor().numpy()
    input_dict = {
        "input": input,
        "tensor1": tensor1,
        "tensor2": tensor2,
        "value": value,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.ones((2, 2, 2)).numpy()
    tensor1 = torch.randn(2, 2, 2).numpy()
    tensor2 = torch.randn(2, 2, 2).numpy()
    value = 0.25
    out = torch.tensor().numpy()
    input_dict = {
        "input": input,
        "tensor1": tensor1,
        "tensor2": tensor2,
        "value": value,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.zeros(4).numpy()
    tensor1 = torch.tensor([8.0, 12.0, 16.0, 20.0]).numpy()
    tensor2 = torch.tensor([4.0, 3.0, 2.0, 5.0]).numpy()
    value = -1.0
    out = torch.tensor().numpy()
    input_dict = {
        "input": input,
        "tensor1": tensor1,
        "tensor2": tensor2,
        "value": value,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([[1.0]]).numpy()
    tensor1 = torch.tensor([[5

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.addcdiv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.addcdiv'.")


check_valid('torch.addcdiv', generated_inputs['torch.addcdiv'], lib="torch", suffix=0)
