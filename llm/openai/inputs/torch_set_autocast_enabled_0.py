
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def set_autocast_enabled_inputs():
    list_of_inputs = []

    input_dict = {"enabled": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"enabled": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"enabled": bool(1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"enabled": bool(0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"enabled": bool(-1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"enabled": bool(2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"enabled": bool("")}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"enabled": bool("enabled")}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"enabled": bool([])}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"enabled": bool([1])}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"enabled": bool({})}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"enabled": bool({"a": 1})}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.set_autocast_enabled"] = set_autocast_enabled_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.set_autocast_enabled' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.set_autocast_enabled'.")


check_valid('torch.set_autocast_enabled', generated_inputs['torch.set_autocast_enabled'], lib="torch", suffix=0)
