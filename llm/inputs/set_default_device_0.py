
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy

def set_default_device_inputs():
    list_of_inputs = []

    input_dict = {
        "device": "cpu"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    if torch.cuda.is_available():
        input_dict = {
            "device": "cuda"
        }
        list_of_inputs.append(copy.deepcopy(input_dict))

        input_dict = {
            "device": "cuda:0"
        }
        list_of_inputs.append(copy.deepcopy(input_dict))

        if torch.cuda.device_count() > 1:
            input_dict = {
                "device": "cuda:1"
            }
            list_of_inputs.append(copy.deepcopy(input_dict))
    
    try:
        if torch.backends.mps.is_available():
            input_dict = {
                "device": "mps"
            }
            list_of_inputs.append(copy.deepcopy(input_dict))
    except:
        pass


    return list_of_inputs

generated_inputs["torch.set_default_device"] = set_default_device_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.set_default_device' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.set_default_device'.")

check_valid('torch.set_default_device', generated_inputs['torch.set_default_device'], lib="torch")
