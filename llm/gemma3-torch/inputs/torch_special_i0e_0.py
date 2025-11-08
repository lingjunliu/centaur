
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def i0e_inputs():
    list_of_inputs = []
    
    input1 = np.array([1.0], dtype=np.float32)
    list_of_inputs.append({"x": torch.tensor(input1)})
    
    return list_of_inputs

generated_inputs["torch.special.i0e"] = i0e_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.special.i0e' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.i0e'.")


check_valid('torch.special.i0e', generated_inputs['torch.special.i0e'], lib="torch", suffix=0)
