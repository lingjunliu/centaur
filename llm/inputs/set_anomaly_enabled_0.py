
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy

def set_anomaly_enabled_inputs():
    list_of_inputs = []

    input_dict = {
        "mode": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "mode": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "mode": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "mode": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "mode": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.set_anomaly_enabled"] = set_anomaly_enabled_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.set_anomaly_enabled' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.set_anomaly_enabled'.")

check_valid('torch.set_anomaly_enabled', generated_inputs['torch.set_anomaly_enabled'], lib="torch")
