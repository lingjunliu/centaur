
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def fold_inputs():
    list_of_inputs = []
    input1 = {
        'output_size': (4, 5),
        'kernel_size': (2, 2),
        'dilation': (1, 1),
        'padding': (0, 0),
        'stride': (1, 1)
    }
    list_of_inputs.append(copy.deepcopy(input1))

    return list_of_inputs

generated_inputs["torch.nn.Fold"] = fold_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.Fold' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Fold'.")


check_valid('torch.nn.Fold', generated_inputs['torch.nn.Fold'], lib="torch", suffix=0)
