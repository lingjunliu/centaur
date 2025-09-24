
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def parse_type_comment_inputs():
    list_of_inputs = []

    # Input 1: Simple comment
    input_dict = {"comment": "torch.Tensor"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Comment with int type
    input_dict = {"comment": "int"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Comment with float type
    input_dict = {"comment": "float"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Comment with bool type
    input_dict = {"comment": "bool"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Comment with str type
    input_dict = {"comment": "str"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.parse_type_comment"] = parse_type_comment_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.parse_type_comment' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.parse_type_comment'.")

check_valid('torch.parse_type_comment', generated_inputs['torch.parse_type_comment'], lib="torch", suffix=0)
