
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def are_deterministic_algorithms_enabled_inputs():
    list_of_inputs = []

    input1 = True
    list_of_inputs.append(copy.deepcopy({"torch.are_deterministic_algorithms_enabled": input1}))

    input2 = False
    list_of_inputs.append(copy.deepcopy({"torch.are_deterministic_algorithms_enabled": input2}))

    input3 = True
    list_of_inputs.append(copy.deepcopy({"torch.are_deterministic_algorithms_enabled": input3}))

    input4 = False
    list_of_inputs.append(copy.deepcopy({"torch.are_deterministic_algorithms_enabled": input4}))

    input5 = True
    list_of_inputs.append(copy.deepcopy({"torch.are_deterministic_algorithms_enabled": input5}))
    
    input6 = False
    list_of_inputs.append(copy.deepcopy({"torch.are_deterministic_algorithms_enabled": input6}))
    
    input7 = True
    list_of_inputs.append(copy.deepcopy({"torch.are_deterministic_algorithms_enabled": input7}))

    input8 = False
    list_of_inputs.append(copy.deepcopy({"torch.are_deterministic_algorithms_enabled": input8}))

    input9 = True
    list_of_inputs.append(copy.deepcopy({"torch.are_deterministic_algorithms_enabled": input9}))

    input10 = False
    list_of_inputs.append(copy.deepcopy({"torch.are_deterministic_algorithms_enabled": input10}))
    
    return list_of_inputs

generated_inputs["torch.are_deterministic_algorithms_enabled"] = are_deterministic_algorithms_enabled_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.are_deterministic_algorithms_enabled' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.are_deterministic_algorithms_enabled'.")


check_valid('torch.are_deterministic_algorithms_enabled', generated_inputs['torch.are_deterministic_algorithms_enabled'], lib="torch", suffix=0)
