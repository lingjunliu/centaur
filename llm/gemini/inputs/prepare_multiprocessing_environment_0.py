
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy

def prepare_multiprocessing_environment_inputs():
    list_of_inputs = []

    input_dict = {
        "file_system": "posix"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "file_system": "nt"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "file_system": "hdfs"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "file_system": "s3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "file_system": ""
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.prepare_multiprocessing_environment"] = prepare_multiprocessing_environment_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.prepare_multiprocessing_environment' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.prepare_multiprocessing_environment'.")

check_valid('torch.prepare_multiprocessing_environment', generated_inputs['torch.prepare_multiprocessing_environment'], lib="torch")
