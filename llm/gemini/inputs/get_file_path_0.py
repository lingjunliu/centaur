
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy

def get_file_path_inputs():
    list_of_inputs = []

    input_dict = {
        "url": "https://example.com/file1.txt"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "url": "http://example.org/file2.zip"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "url": "ftp://example.net/file3.tar.gz"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "url": "https://www.example.com/file4.pdf"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "url": "https://subdomain.example.com/file5.jpg"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "url": "https://example.com/path/to/file6.html"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.get_file_path"] = get_file_path_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.get_file_path' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.get_file_path'.")

check_valid('torch.get_file_path', generated_inputs['torch.get_file_path'], lib="torch")
