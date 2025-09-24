
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy

def parse_schema_inputs():
    list_of_inputs = []
    
    input_dict = {
        "schema_string": "aten::add.Scalar(Tensor self, Scalar other, Scalar alpha) -> Tensor"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "schema_string": "aten::abs(Tensor self) -> Tensor"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "schema_string": "aten::zeros(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "schema_string": "aten::rand(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "schema_string": "aten::empty(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.parse_schema"] = parse_schema_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.parse_schema' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.parse_schema'.")

check_valid('torch.parse_schema', generated_inputs['torch.parse_schema'], lib="torch")
