
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy
import numpy as np

def dequantize_inputs():
    list_of_inputs = []

    qint8_tensor = torch.quantize_per_tensor(torch.randn(3, 4), 0.5, 10, torch.quint8)
    input_dict = {"tensors": [qint8_tensor.numpy()]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    quint8_tensor = torch.quantize_per_tensor(torch.rand(2, 2), 0.2, 5, torch.quint8)
    input_dict = {"tensors": [quint8_tensor.numpy()]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    int32_tensor = (torch.randn(1, 5) * 100).int()
    scale = 1.0
    zero_point = 0
    
    quantized_tensor = torch.quantize_per_tensor(int32_tensor.float(), scale=scale, zero_point=zero_point, q_dtype=torch.qint32)
    input_dict = {"tensors": [quantized_tensor.numpy()]}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    qint8_tensor_2d = torch.quantize_per_tensor(torch.randn(3, 4), 0.5, 10, torch.quint8)
    quint8_tensor_2d = torch.quantize_per_tensor(torch.rand(2, 2), 0.2, 5, torch.quint8)
    input_dict = {"tensors": [qint8_tensor_2d.numpy(), quint8_tensor_2d.numpy()]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    qint8_tensor_3d = torch.quantize_per_tensor(torch.randn(3, 4, 5), 0.5, 10, torch.quint8)
    input_dict = {"tensors": [qint8_tensor_3d.numpy()]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.dequantize_2"] = dequantize_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.dequantize_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.dequantize_2'.")

check_valid('torch.dequantize', generated_inputs['torch.dequantize_2'], lib="torch")
