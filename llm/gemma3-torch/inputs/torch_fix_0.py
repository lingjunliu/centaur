
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def fix_inputs():
    list_of_inputs = []

    input1 = torch.tensor([1.5, 2.7, 3.1]).numpy()
    out1 = torch.tensor([]) # Corrected: Initialize tensor with no data
    input_dict1 = {"input": input1, "out": out1.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.tensor([-1.5, -2.7, -3.1]).numpy()
    out2 = torch.tensor([])
    input_dict2 = {"input": input2, "out": out2.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.tensor([1.0, -2.0, 3.0, -4.0]).numpy()
    out3 = torch.tensor([])
    input_dict3 = {"input": input3, "out": out3.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(2, 3).numpy()
    out4 = torch.tensor([])
    input_dict4 = {"input": input4, "out": out4.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(3, 2, 4).numpy()
    out5 = torch.tensor([])
    input_dict5 = {"input": input5, "out": out5.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.fix"] = fix_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.fix' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.fix'.")


check_valid('torch.fix', generated_inputs['torch.fix'], lib="torch", suffix=0)
