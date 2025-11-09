
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import torch
import copy
import numpy as np

def dequantize_inputs():
    list_of_inputs = []
    
    x = torch.quantize_per_tensor(torch.tensor([1.0, 2.0, 3.0]), scale=0.1, zero_point=0, dtype=torch.quint8)
    tensors = [x]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = torch.quantize_per_tensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), scale=0.5, zero_point=128, dtype=torch.quint8)
    tensors = [x]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x1 = torch.quantize_per_tensor(torch.tensor([1.0, 2.0, 3.0]), scale=0.1, zero_point=0, dtype=torch.quint8)
    x2 = torch.quantize_per_tensor(torch.tensor([4.0, 5.0, 6.0]), scale=0.2, zero_point=10, dtype=torch.quint8)
    tensors = [x1, x2]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = torch.quantize_per_tensor(torch.randn(2, 3, 4), scale=1.0, zero_point=0, dtype=torch.quint8)
    tensors = [x]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = torch.quantize_per_tensor(torch.tensor([-1.0, 0.0, 1.0]), scale=0.05, zero_point=0, dtype=torch.qint8)
    tensors = [x]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = torch.quantize_per_tensor(torch.tensor([10.0, 20.0, 30.0]), scale=0.01, zero_point=0, dtype=torch.qint32)
    tensors = [x]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = torch.quantize_per_tensor(torch.tensor([100.0, 200.0, 300.0]), scale=10.0, zero_point=0, dtype=torch.quint8)
    tensors = [x]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x1 = torch.quantize_per_tensor(torch.ones(5), scale=0.5, zero_point=0, dtype=torch.quint8)
    x2 = torch.quantize_per_tensor(torch.zeros(3, 3), scale=0.1, zero_point=128, dtype=torch.quint8)
    tensors = [x1, x2]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = torch.quantize_per_tensor(torch.randn(1, 3, 8, 8), scale=0.3, zero_point=0, dtype=torch.quint8)
    tensors = [x]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = torch.quantize_per_tensor(torch.tensor([0.5, 1.5, 2.5]), scale=0.25, zero_point=64, dtype=torch.qu

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.dequantize_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.dequantize_2'.")


check_valid('torch.dequantize', generated_inputs['torch.dequantize_2'], lib="torch", suffix=2)
