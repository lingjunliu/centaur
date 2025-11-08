
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def i1e_inputs():
    list_of_inputs = []
    
    input1 = torch.tensor([1.0]).numpy()
    list_of_inputs.append({"x": input1})
    
    input2 = torch.tensor([-1.0, 2.0]).numpy()
    list_of_inputs.append({"x": input2})
    
    input3 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).numpy()
    list_of_inputs.append({"x": input3})
    
    input4 = torch.tensor([1.0, -2.0, 3.0, -4.0]).numpy()
    list_of_inputs.append({"x": input4})
    
    input5 = torch.tensor([0.0]).numpy()
    list_of_inputs.append({"x": input5})
    
    input6 = torch.tensor([1e-9]).numpy()
    list_of_inputs.append({"x": input6})

    input7 = torch.tensor([1e9]).numpy()
    list_of_inputs.append({"x": input7})

    input8 = torch.tensor([1.0, 2.0]).numpy()
    list_of_inputs.append({"x": input8})
    
    input9 = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    list_of_inputs.append({"x": input9})
    
    input10 = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).numpy()
    list_of_inputs.append({"x": input10})
    
    input11 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]).numpy()
    list_of_inputs.append({"x": input11})

    return list_of_inputs

generated_inputs["torch.special.i1e"] = i1e_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.special.i1e' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.i1e'.")


check_valid('torch.special.i1e', generated_inputs['torch.special.i1e'], lib="torch", suffix=0)
