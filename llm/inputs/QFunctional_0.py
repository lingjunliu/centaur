
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np
import torch.nn.quantized as nnq

def qfunctional_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = torch.randn(3, 4, dtype=torch.float32)
    scale = 0.5
    zero_point = 0
    q_tensor = torch.quantize_per_tensor(input_tensor, scale=scale, zero_point=zero_point, dtype=torch.qint8)
    q_functional = nnq.QFunctional()
    input_dict = {"x": q_tensor, "self": q_functional}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = torch.randn(2, 2, dtype=torch.float32)
    scale = 0.1
    zero_point = 10
    q_tensor = torch.quantize_per_tensor(input_tensor, scale=scale, zero_point=zero_point, dtype=torch.qint8)
    q_functional = nnq.QFunctional()
    input_dict = {"x": q_tensor, "self": q_functional}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = torch.randn(1, 5, dtype=torch.float32)
    scale = 1.0
    zero_point = -5
    q_tensor = torch.quantize_per_tensor(input_tensor, scale=scale, zero_point=zero_point, dtype=torch.qint8)
    q_functional = nnq.QFunctional()
    input_dict = {"x": q_tensor, "self": q_functional}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = torch.randn(4, 1, dtype=torch.float32)
    scale = 0.25
    zero_point = 2
    q_tensor = torch.quantize_per_tensor(input_tensor, scale=scale, zero_point=zero_point, dtype=torch.qint8)
    q_functional = nnq.QFunctional()
    input_dict = {"x": q_tensor, "self": q_functional}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    input_tensor = torch.randn(2, 3, 4, dtype=torch.float32)
    scale = 0.3
    zero_point = -1
    q_tensor = torch.quantize_per_tensor(input_tensor, scale=scale, zero_point=zero_point, dtype=torch.qint8)
    q_functional = nnq.QFunctional()
    input_dict = {"x": q_tensor, "self": q_functional}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = torch.randn(size=(1,), dtype=torch.float32)
    scale = 0.7
    zero_point = 3
    q_tensor = torch.quantize_per_tensor(input_tensor, scale=scale, zero_point=zero_point, dtype=torch.qint8)
    q_functional = nnq.QFunctional()
    input_dict = {"x": q_tensor, "self": q_functional}
    list_of_inputs.append(copy.deepcopy(input_dict))
    

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.quantized.QFunctional"] = qfunctional_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.quantized.QFunctional' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.quantized.QFunctional'.")

check_valid('torch.nn.quantized.QFunctional', generated_inputs['torch.nn.quantized.QFunctional'], lib="torch", suffix=0)
