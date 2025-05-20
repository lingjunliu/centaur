
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy
from typing import List, Dict, Tuple, Optional

def isinstance_inputs():
    list_of_inputs = []

    # Test case 1: List of ints
    obj1 = [1, 2, 3]
    target_type1 = List[int]
    input_dict1 = {"obj": obj1, "target_type": target_type1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Test case 2: Single integer
    obj2 = 5
    target_type2 = int
    input_dict2 = {"obj": obj2, "target_type": target_type2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Test case 3: String
    obj3 = "hello"
    target_type3 = str
    input_dict3 = {"obj": obj3, "target_type": target_type3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Test case 4: Boolean
    obj4 = True
    target_type4 = bool
    input_dict4 = {"obj": obj4, "target_type": target_type4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Test case 5: Tuple of ints
    obj5 = (1, 2)
    target_type5 = Tuple[int, int]
    input_dict5 = {"obj": obj5, "target_type": target_type5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

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
