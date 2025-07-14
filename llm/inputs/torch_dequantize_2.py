
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def dequantize_inputs():
    list_of_inputs = []

    # Input 1: Empty list
    tensors = []
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Single quantized tensor
    qtensor = torch.quantize_per_tensor(torch.tensor([1.0, 2.0, 3.0]), scale=torch.tensor(0.5), zero_point=torch.tensor(0), dtype=torch.qint8)
    tensors = [qtensor]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: List of quantized tensors with different shapes
    qtensor1 = torch.quantize_per_tensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), scale=torch.tensor(0.5), zero_point=torch.tensor(0), dtype=torch.qint8)
    qtensor2 = torch.quantize_per_tensor(torch.tensor([5.0, 6.0]), scale=torch.tensor(0.25), zero_point=torch.tensor(0), dtype=torch.qint8)
    tensors = [qtensor1, qtensor2]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: List of quantized tensors with different dtypes
    qtensor1 = torch.quantize_per_tensor(torch.tensor([1.0, 2.0]), scale=torch.tensor(0.5), zero_point=torch.tensor(0), dtype=torch.qint8)
    qtensor2 = torch.quantize_per_tensor(torch.tensor([3.0, 4.0]), scale=torch.tensor(0.25), zero_point=torch.tensor(128), dtype=torch.quint8)
    tensors = [qtensor1, qtensor2]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D tensor
    qtensor = torch.quantize_per_tensor(torch.randn(2, 3, 4), scale=torch.tensor(0.1), zero_point=torch.tensor(0), dtype=torch.qint8)
    tensors = [qtensor]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Quantized tensor with negative values
    qtensor = torch.quantize_per_tensor(torch.tensor([-1.0, -2.0, 3.0]), scale=torch.tensor(0.5), zero_point=torch.tensor(0), dtype=torch.qint8)
    tensors = [qtensor]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Multiple tensors with different scales and zero points
    qtensor1 = torch.quantize_per_tensor(torch.tensor([1.0, 2.0]), scale=torch.tensor(0.5), zero_point=torch.tensor(0), dtype=torch.qint8)
    qtensor2 = torch.quantize_per_tensor(torch.tensor([3.0, 4.0]), scale=torch.tensor(0.25), zero_point=torch.tensor(128), dtype=torch.quint8)
    tensors = [qtensor1, qtensor2]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Larger tensors
    qtensor1 = torch.quantize_per_tensor(torch.randn(100), scale=torch.tensor(0.1), zero_point=torch.tensor(0), dtype=torch.qint8)
    qtensor2 = torch.quantize_per_tensor(torch.randn(50, 2), scale=torch.tensor(0.05), zero_point=torch.tensor(128), dtype=torch.quint8)
    tensors = [qtensor1, qtensor2]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Tensors with zero scale
    qtensor1 = torch.quantize_per_tensor(torch.tensor([1.0, 2.0]), scale=torch.tensor(0.00001), zero_point=torch.tensor(0), dtype=torch.qint8)
    qtensor2 = torch.quantize_per_tensor(torch.tensor([3.0, 4.0]), scale=torch.tensor(0.000001), zero_point=torch.tensor(128), dtype=torch.quint8)
    tensors = [qtensor1, qtensor2]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: List of different dimensions
    qtensor1 = torch.quantize_per_tensor(torch.randn(1), scale=torch.tensor(0.1), zero_point=torch.tensor(0), dtype=torch.qint8)
    qtensor2 = torch.quantize_per_tensor(torch.randn(1,1), scale=torch.tensor(0.05), zero_point=torch.tensor(128), dtype=torch.quint8)
    qtensor3 = torch.quantize_per_tensor(torch.randn(1,1,1), scale=torch.tensor(0.05), zero_point=torch.tensor(0), dtype=torch.qint8)
    tensors = [qtensor1, qtensor2, qtensor3]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.dequantize_2"] = dequantize_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.dequantize_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.dequantize_2'.")

check_valid('torch.dequantize', generated_inputs['torch.dequantize_2'], lib="torch", suffix=2)
