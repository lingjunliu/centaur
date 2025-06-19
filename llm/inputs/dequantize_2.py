
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def dequantize_inputs():
    list_of_inputs = []

    def quantize(tensor, scale, zero_point, dtype):
        qtensor = torch.quantize_per_tensor(tensor, scale, zero_point, dtype)
        return qtensor

    # Input 1: Empty list
    tensors = []
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Single quantized tensor
    qtensor = quantize(torch.randn(5), 0.5, 10, torch.qint8)
    tensors = [qtensor.dequantize().numpy()]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multiple quantized tensors
    qtensor1 = quantize(torch.randn(2, 3), 0.2, 5, torch.qint8)
    qtensor2 = quantize(torch.randn(3, 2), 0.3, 8, torch.qint8)
    tensors = [qtensor1.dequantize().numpy(), qtensor2.dequantize().numpy()]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Quantized tensors with different scales and zero points
    qtensor1 = quantize(torch.randn(4, 4), 0.1, 0, torch.qint8)
    qtensor2 = quantize(torch.randn(4, 4), 0.5, 10, torch.qint8)
    tensors = [qtensor1.dequantize().numpy(), qtensor2.dequantize().numpy()]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Quantized tensor with negative values
    qtensor = quantize(torch.randn(1, 5) * -1, 0.25, -5, torch.qint8)
    tensors = [qtensor.dequantize().numpy()]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.dequantize_2"] = dequantize_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.dequantize_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.dequantize_2'.")

check_valid('torch.dequantize', generated_inputs['torch.dequantize_2'], lib="torch", suffix=2)
