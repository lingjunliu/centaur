
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy

def ScriptWarning_inputs():
    list_of_inputs = []

    input_dict = {
        "msg": "This is a simple warning message."
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "msg": "Warning: Using a deprecated feature."
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "msg": "Potential type mismatch detected."
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "msg": "Unexpected behavior might occur."
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "msg": "Performance degradation possible."
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "msg": ""
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "msg": "This is a very long warning message that might wrap around the screen, testing how long messages are handled."
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.jit.ScriptWarning"] = ScriptWarning_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.jit.ScriptWarning' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.jit.ScriptWarning'.")

check_valid('torch.jit.ScriptWarning', generated_inputs['torch.jit.ScriptWarning'], lib="torch")
