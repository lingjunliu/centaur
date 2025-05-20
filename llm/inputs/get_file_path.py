
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy

def get_file_path_inputs():
    list_of_inputs = []

    input_dict = {
        "url": "https://example.com/model1.pth",
        "model_dir": "./models",
        "progress": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "url": "http://example.org/data.tar.gz",
        "model_dir": "/tmp/downloads",
        "progress": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "url": "ftp://example.net/weights.pt",
        "model_dir": "~/torch_models",
        "progress": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "url": "https://example.com/large_model.bin",
        "model_dir": ".",
        "progress": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "url": "https://my.domain.com/another_model.dat",
        "model_dir": "my_models",
        "progress": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = get_file_path_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('get_file_path', generated_inputs)
