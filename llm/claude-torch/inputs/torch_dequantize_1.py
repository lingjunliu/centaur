
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import torch
import numpy as np
import copy

def dequantize_inputs():
    list_of_inputs = []
    
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    scale = 0.1
    zero_point = 0
    q_tensor = torch.quantize_per_tensor(x, scale, zero_point, torch.qint8)
    input_dict = {"tensor": q_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = torch.randn(3, 4)
    scale = 0.5
    zero_point = 10
    q_tensor = torch.quantize_per_tensor(x, scale, zero_point, torch.qint8)
    input_dict = {"tensor": q_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = torch.randn(2, 3, 4)
    scale = 0.2
    zero_point = 5
    q_tensor = torch.quantize_per_tensor(x, scale, zero_point, torch.qint8)
    input_dict = {"tensor": q_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = torch.tensor([-5.0, -2.0, 0.0, 2.0, 5.0])
    scale = 0.05
    zero_point = 0
    q_tensor = torch.quantize_per_tensor(x, scale, zero_point, torch.qint8)
    input_dict = {"tensor": q_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = torch.randn(2, 3, 4, 5)
    scale = 0.3
    zero_point = 20
    q_tensor = torch.quantize_per_tensor(x, scale, zero_point, torch.qint8)
    input_dict = {"tensor": q_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = torch.tensor([42.0])
    scale = 1.0
    zero_point = 0
    q_tensor = torch.quantize_per_tensor(x, scale, zero_point, torch.qint8)
    input_dict = {"tensor": q_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = torch.tensor([0.5, 1.5, 2.5, 3.5])
    scale = 0.1
    zero_point = 128
    q_tensor = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
    input_dict = {"tensor": q_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = torch.randn(10, 10)
    scale = 0.15
    zero_point = 15
    q_tensor = torch.quantize_per_tensor(x, scale, zero_point, torch.qint8)
    input_dict = {"tensor": q_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = torch.tensor([100.0, 200.0, 300.0])
    scale = 0.01
    zero_point = 0
    q_tensor = torch.quantize_per_tensor(x, scale, zero_point, torch.qint8)
    input_dict = {"tensor": q_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = torch.ones(5, 5)
    scale = 0.25
    zero_point = 50
    q_tensor = torch.quantize_per_tensor(x, scale, zero_point, torch.qint8)
    input_dict = {"tensor": q_tensor}
    list_of_inputs.append(copy.deepc

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.dequantize_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.dequantize_1'.")


check_valid('torch.dequantize', generated_inputs['torch.dequantize_1'], lib="torch", suffix=1)
