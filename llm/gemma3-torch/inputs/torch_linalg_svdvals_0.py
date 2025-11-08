
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def svdvals_inputs():
    list_of_inputs = []
    
    input1 = torch.randn(5, 3).numpy()
    driver1 = None
    out1 = np.zeros((3,), dtype=np.float32)
    
    input_dict1 = {
        "A": input1,
        "driver": driver1,
        "out": out1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = torch.randn(2, 2).numpy()
    driver2 = None
    out2 = np.zeros((2,), dtype=np.float32)
    
    input_dict2 = {
        "A": input2,
        "driver": driver2,
        "out": out2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(3, 5).numpy()
    driver3 = None
    out3 = np.zeros((5,), dtype=np.float32)
    
    input_dict3 = {
        "A": input3,
        "driver": driver3,
        "out": out3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = torch.randn(4, 4).numpy()
    driver4 = None
    out4 = np.zeros((4,), dtype=np.float32)
    
    input_dict4 = {
        "A": input4,
        "driver": driver4,
        "out": out4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = torch.randn(1, 1).numpy()
    driver5 = None
    out5 = np.zeros((1,), dtype=np.float32)
    
    input_dict5 = {
        "A": input5,
        "driver": driver5,
        "out": out5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.linalg.svdvals"] = svdvals_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.linalg.svdvals' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.svdvals'.")


check_valid('torch.linalg.svdvals', generated_inputs['torch.linalg.svdvals'], lib="torch", suffix=0)
