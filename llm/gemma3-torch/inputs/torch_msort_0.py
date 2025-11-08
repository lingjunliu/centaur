
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def msort_inputs():
    list_of_inputs = []
    
    input1 = torch.randn(3, 4).numpy()
    out1 = torch.empty(3, 4).numpy()
    input_dict1 = {"input": input1, "out": out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = torch.tensor([1, 5, 2, 4, 3]).numpy()
    out2 = torch.empty(5).numpy()
    input_dict2 = {"input": input2, "out": out2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    return list_of_inputs

generated_inputs["torch.msort"] = msort_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.msort' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.msort'.")


check_valid('torch.msort', generated_inputs['torch.msort'], lib="torch", suffix=0)
