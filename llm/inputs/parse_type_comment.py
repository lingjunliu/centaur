
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy

def parse_type_comment_inputs():
    list_of_inputs = []

    # Example 1: Simple type hint
    input_dict = {
        "comment": "Tensor",
        "module": "torch"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs

generated_inputs = parse_type_comment_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('parse_type_comment', generated_inputs)
