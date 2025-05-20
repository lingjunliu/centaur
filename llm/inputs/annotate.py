
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np
from typing import Dict, List, Optional

def annotate_inputs():
    list_of_inputs = []

    # Example 1: Empty dictionary
    input_dict = {
        "the_type": "Dict[str, int]",
        "the_value": {}
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2: Optional type with a value
    input_dict = {
        "the_type": "Optional[int]",
        "the_value": 10
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 3: Empty list
    input_dict = {
        "the_type": "List[float]",
        "the_value": []
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 4: List of tensors
    input_dict = {
        "the_type": "List[torch.Tensor]",
        "the_value": [torch.randn(2, 3).numpy(), torch.randn(4, 5).numpy()]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 5: Dictionary of tensors
    input_dict = {
        "the_type": "Dict[str, torch.Tensor]",
        "the_value": {"a": torch.randn(2, 2).numpy(), "b": torch.randn(3, 3).numpy()}
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 6: Optional tensor
    input_dict = {
        "the_type": "Optional[torch.Tensor]",
        "the_value": torch.randn(1, 1).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 7: None value for optional tensor
    input_dict = {
        "the_type": "Optional[torch.Tensor]",
        "the_value": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 8: List of ints
    input_dict = {
        "the_type": "List[int]",
        "the_value": [1, 2, 3, 4, 5]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Example 9: Dict of string and float
    input_dict = {
        "the_type": "Dict[str, float]",
        "the_value": {"key1": 1.0, "key2": 2.0}
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = annotate_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('annotate', generated_inputs)
