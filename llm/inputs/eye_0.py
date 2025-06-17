
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy
import numpy as np

def torch_eye_inputs():
    list_of_inputs = []

    # Case 1: Basic case with only n
    input_dict = {
        "n": 3,
        "m": None,
        "out": None,
        "dtype": None,
        "layout": "strided",
        "device": None,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Different n and m
    input_dict = {
        "n": 2,
        "m": 5,
        "out": None,
        "dtype": None,
        "layout": "strided",
        "device": None,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Specifying dtype
    input_dict = {
        "n": 4,
        "m": None,
        "out": None,
        "dtype": torch.float64,
        "layout": "strided",
        "device": None,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Requires grad true
    input_dict = {
        "n": 3,
        "m": None,
        "out": None,
        "dtype": None,
        "layout": "strided",
        "device": None,
        "requires_grad": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}

def convert_to_torch_format(input_dict):
    args = []
    kwargs = {}

    args.append(input_dict['n'])
    if input_dict['m'] is not None:
        args.append(input_dict['m'])
    
    if input_dict['out'] is not None:
        kwargs['out'] = torch.tensor(input_dict['out'])
    if input_dict['dtype'] is not None:
        kwargs['dtype'] = input_dict['dtype']
    if input_dict['layout'] == "strided":
        kwargs['layout'] = torch.strided
    if input_dict['device'] is not None:
        kwargs['device'] = input_dict['device']
    kwargs['requires_grad'] = input_dict['requires_grad']

    return {"args": args, "kwargs": kwargs}

api_input_list = torch_eye_inputs()
processed_input_list = []
for input_dict in api_input_list:
    processed_input_list.append(convert_to_torch_format(input_dict))

generated_inputs["torch.eye"] = processed_input_list

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.eye' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.eye'.")

check_valid('torch.eye', generated_inputs['torch.eye'], lib="torch")
