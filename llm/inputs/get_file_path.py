
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def get_file_path_inputs():
    list_of_inputs = []
    
    input_dict = {
        "url": "https://example.com/model1.pth",
        "model_dir": "./models",
        "progress": True,
        "input": np.array([1.0]) # Added to avoid TypeError, using numpy array
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "url": "http://example.org/data.tar.gz",
        "model_dir": "/tmp/data",
        "progress": False,
        "input": np.array([1.0]) # Added to avoid TypeError, using numpy array
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "url": "ftp://example.net/weights.pt",
        "model_dir": "~/weights",
        "progress": True,
        "input": np.array([1.0]) # Added to avoid TypeError, using numpy array
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "url": "https://s3.amazonaws.com/bucket/model.bin",
        "model_dir": "C:\\Models",
        "progress": False,
        "input": np.array([1.0]) # Added to avoid TypeError, using numpy array
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "url": "https://huggingface.co/model.safetensors",
        "model_dir": "./cache",
        "progress": True,
        "input": np.array([1.0]) # Added to avoid TypeError, using numpy array
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "url": "https://github.com/user/repo/releases/download/v1.0/model.zip",
        "model_dir": "/opt/models",
        "progress": False,
        "input": np.array([1.0]) # Added to avoid TypeError, using numpy array
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "url": "https://www.dropbox.com/s/randomstring/model.ckpt?dl=1",
        "model_dir": "/usr/local/models",
        "progress": True,
        "input": np.array([1.0]) # Added to avoid TypeError, using numpy array
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "url": "file:///home/user/local_model.pt",
        "model_dir": ".",
        "progress": False,
        "input": np.array([1.0]) # Added to avoid TypeError, using numpy array
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
