
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy

def parse_type_comment_inputs():
    list_of_inputs = []

    input_dict = {
        "comment": "Sequence[Tensor]"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.parse_type_comment"] = parse_type_comment_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.parse_type_comment' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.parse_type_comment'.")

check_valid('torch.parse_type_comment', generated_inputs['torch.parse_type_comment'], lib="torch")
