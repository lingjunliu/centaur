
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy
import numpy as np
from typing import List, Dict, Tuple, Optional

def isinstance_inputs():
    list_of_inputs = []

    obj1 = [torch.randn(3, 3).numpy(), torch.randn(4, 4).numpy()]
    target_type1 = List[torch.Tensor]
    input_dict1 = {"obj": obj1, "target_type": target_type1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    obj2 = {"key1": "val1", "key2": "val2"}
    target_type2 = Dict[str, str]
    input_dict2 = {"obj": obj2, "target_type": target_type2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    return list_of_inputs

generated_inputs = isinstance_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('isinstance', generated_inputs)
