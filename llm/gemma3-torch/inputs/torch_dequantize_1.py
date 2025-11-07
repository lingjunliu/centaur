
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def dequantize_inputs():
    list_of_inputs = []
    
    input1 = torch.randn(3)
    list_of_inputs.append({"tensor": input1.numpy()})

    input2 = torch.randn(2, 4)
    list_of_inputs.append({"tensor": input2.numpy()})

    input3 = torch.randn(5, 5, 5)
    list_of_inputs.append({"tensor": input3.numpy()})

    input4 = torch.randn(2, 2)
    list_of_inputs.append({"tensor": input4.numpy()})

    input5 = torch.randn(4)
    list_of_inputs.append({"tensor": input5.numpy()})
    
    input6 = torch.randn(10, 10, 10, 10)
    list_of_inputs.append({"tensor": input6.numpy()})

    input7 = torch.randn(3, 3)
    list_of_inputs.append({"tensor": input7.numpy()})
    
    input8 = torch.randn(2, 3, 4)
    list_of_inputs.append({"tensor": input8.numpy()})

    input9 = torch.randn(6, 6)
    list_of_inputs.append({"tensor": input9.numpy()})

    input10 = torch.randn(7)
    list_of_inputs.append({"tensor": input10.numpy()})
    
    return list_of_inputs

generated_inputs["torch.dequantize_1"] = dequantize_inputs()

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
