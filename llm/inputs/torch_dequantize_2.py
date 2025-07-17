
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
    qtensor = torch.quantize_per_tensor(torch.tensor([1.0, 2.0, 3.0]), 0.5, 10, torch.quint8)
    tensors = [qtensor]
    input_dict = {"tensors": [t.int_repr().numpy() for t in tensors]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multiple quantized tensors
    qtensor1 = torch.quantize_per_tensor(torch.tensor([1.0, 2.0, 3.0]), 0.5, 10, torch.quint8)
    qtensor2 = torch.quantize_per_tensor(torch.tensor([4.0, 5.0, 6.0]), 0.25, 5, torch.quint8)
    tensors = [qtensor1, qtensor2]
    input_dict = {"tensors": [t.int_repr().numpy() for t in tensors]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Quantized tensor with different dtype (qint8)
    qtensor = torch.quantize_per_tensor(torch.tensor([-1.0, -2.0, -3.0]), 0.5, 0, torch.qint8)
    tensors = [qtensor]
    input_dict = {"tensors": [t.int_repr().numpy() for t in tensors]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Quantized 2D tensor
    qtensor = torch.quantize_per_tensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), 0.5, 10, torch.quint8)
    tensors = [qtensor]
    input_dict = {"tensors": [t.int_repr().numpy() for t in tensors]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Mixed dtypes (quint8 and qint8)
    qtensor1 = torch.quantize_per_tensor(torch.tensor([1.0, 2.0, 3.0]), 0.5, 10, torch.quint8)
    qtensor2 = torch.quantize_per_tensor(torch.tensor([-4.0, -5.0, -6.0]), 0.25, 0, torch.qint8)
    tensors = [qtensor1, qtensor2]
    input_dict = {"tensors": [t.int_repr().numpy() for t in tensors]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different scales and zero points
    qtensor1 = torch.quantize_per_tensor(torch.tensor([1.0, 2.0, 3.0]), 0.1, 5, torch.quint8)
    qtensor2 = torch.quantize_per_tensor(torch.tensor([4.0, 5.0, 6.0]), 0.75, 20, torch.quint8)
    tensors = [qtensor1, qtensor2]
    input_dict = {"tensors": [t.int_repr().numpy() for t in tensors]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Large values
    qtensor = torch.quantize_per_tensor(torch.tensor([100.0, 200.0, 300.0]), 5.0, 10, torch.quint8)
    tensors = [qtensor]
    input_dict = {"tensors": [t.int_repr().numpy() for t in tensors]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Negative values and zero point not 0
    qtensor = torch.quantize_per_tensor(torch.tensor([-1.0, 0.0, 1.0]), 0.2, 100, torch.quint8)
    tensors = [qtensor]
    input_dict = {"tensors": [t.int_repr().numpy() for t in tensors]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D tensor
    qtensor = torch.quantize_per_tensor(torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]), 0.5, 10, torch.quint8)
    tensors = [qtensor]
    input_dict = {"tensors": [t.int_repr().numpy() for t in tensors]}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: List with an empty quantized tensor representation
    qtensor = torch.empty(0).to(torch.quint8)  # Create an empty quantized tensor

    if qtensor.numel() == 0:
        qtensor = torch.quantize_per_tensor(torch.tensor([]), 1.0, 0, torch.quint8)

    tensors = [qtensor]
    input_dict = {"tensors": [t.int_repr().numpy() for t in tensors]}
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
