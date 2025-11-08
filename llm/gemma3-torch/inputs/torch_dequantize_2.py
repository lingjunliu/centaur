
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def dequantize_inputs():
    list_of_inputs = []
    
    tensor1 = np.array([1, 2, 3], dtype=np.int8)
    list_of_inputs.append([tensor1])
    
    return list_of_inputs

generated_inputs["torch.dequantize_2"] = dequantize_inputs()

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
