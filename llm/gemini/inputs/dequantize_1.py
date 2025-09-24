
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def dequantize_inputs():
    list_of_inputs = []

    qint8_tensor = torch.quantize_per_tensor(torch.randn(3, 4), 0.5, 10, torch.quint8)
    input_dict = {"tensor": qint8_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    quint8_tensor = torch.quantize_per_tensor(torch.rand(2, 2), 0.2, 5, torch.quint8)
    input_dict = {"tensor": quint8_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    qint32_tensor = torch.quantize_per_tensor(torch.randn(1, 5, 5), 0.1, 0, torch.quint8)
    input_dict = {"tensor": qint32_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    qint4_tensor = torch.quantize_per_tensor(torch.randn(4, 1), 0.75, 5, torch.quint8)
    input_dict = {"tensor": qint4_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    quint4_tensor = torch.quantize_per_tensor(torch.rand(3, 2, 1), 0.3, 8, torch.quint8)
    input_dict = {"tensor": quint4_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.dequantize_1"] = dequantize_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.dequantize_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.dequantize_1'.")

check_valid('torch.dequantize', generated_inputs['torch.dequantize_1'], lib="torch")
