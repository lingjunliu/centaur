
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy
from typing import List, Dict, Tuple, Optional

def torch_jit_isinstance_inputs():
    list_of_inputs = []

    # Example 1: List[torch.Tensor]
    obj1 = [torch.randn(3, 3).numpy(), torch.randn(4, 3).numpy()]
    target_type1 = List[torch.Tensor]
    input_dict1 = {"obj": obj1, "target_type": target_type1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Example 2: Dict[str, str]
    obj2 = {"key1": "val1", "key2": "val2"}
    target_type2 = Dict[str, str]
    input_dict2 = {"obj": obj2, "target_type": target_type2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Example 3: Optional[Tuple[int, str, int]]
    obj3 = (1, "hello", 2)
    target_type3 = Optional[Tuple[int, str, int]]
    input_dict3 = {"obj": obj3, "target_type": target_type3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Example 4: List[int]
    obj4 = [1, 2, 3, 4, 5]
    target_type4 = List[int]
    input_dict4 = {"obj": obj4, "target_type": target_type4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Example 5: Tuple[float, float, float]
    obj5 = (1.0, 2.5, 3.7)
    target_type5 = Tuple[float, float, float]
    input_dict5 = {"obj": obj5, "target_type": target_type5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Example 6: Dict[str, List[torch.Tensor]]
    obj6 = {"key1": [torch.randn(2, 2).numpy(), torch.randn(1, 1).numpy()], "key2": [torch.randn(3, 3).numpy()]}
    target_type6 = Dict[str, List[torch.Tensor]]
    input_dict6 = {"obj": obj6, "target_type": target_type6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Example 7: Optional[int]
    obj7 = 5
    target_type7 = Optional[int]
    input_dict7 = {"obj": obj7, "target_type": target_type7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.jit.isinstance"] = torch_jit_isinstance_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.jit.isinstance' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.jit.isinstance'.")

check_valid('torch.jit.isinstance', generated_inputs['torch.jit.isinstance'], lib="torch")
