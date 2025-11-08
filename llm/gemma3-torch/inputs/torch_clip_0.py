
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def clip_inputs():
    list_of_inputs = []
    
    input1 = np.array([1.0, -2.0, 3.0])
    min1 = np.array([0.0, -1.0, 2.0])
    max1 = np.array([2.0, 0.0, 4.0])
    out1 = np.array([])

    input_dict1 = {
        "input": input1,
        "min": min1,
        "max": max1,
        "out": out1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.random.rand(2, 3)
    min2 = np.array([-1.0])
    max2 = np.array([1.0])
    out2 = np.array([])
    
    input_dict2 = {
        "input": input2,
        "min": min2,
        "max": max2,
        "out": out2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    return list_of_inputs

generated_inputs["torch.clip"] = clip_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.clip' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.clip'.")


check_valid('torch.clip', generated_inputs['torch.clip'], lib="torch", suffix=0)
