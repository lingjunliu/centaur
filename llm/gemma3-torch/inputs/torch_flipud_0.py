
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def flipud_inputs():
    list_of_inputs = []
    
    input1 = torch.arange(4).view(2, 2).numpy()
    input_dict = {"input": input1}
    list_of_inputs.append(input_dict)

    input2 = torch.tensor([[1, 2, 3], [4, 5, 6]]).numpy()
    input_dict = {"input": input2}
    list_of_inputs.append(input_dict)
    
    input3 = torch.randn(3, 4).numpy()
    input_dict = {"input": input3}
    list_of_inputs.append(input_dict)
    
    return list_of_inputs

generated_inputs["torch.flipud"] = flipud_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.flipud' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.flipud'.")


check_valid('torch.flipud', generated_inputs['torch.flipud'], lib="torch", suffix=0)
