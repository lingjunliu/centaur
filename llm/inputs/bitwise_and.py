
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def bitwise_and_inputs():
    list_of_inputs = []

    input1 = torch.tensor([1, 2, 3, 4], dtype=torch.int32).numpy()
    input2 = torch.tensor([0, 2, 5, 8], dtype=torch.int32).numpy()
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = torch.tensor([[1, 2], [3, 4]], dtype=torch.uint8).numpy()
    input2 = torch.tensor([[0, 2], [5, 0]], dtype=torch.uint8).numpy()
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = torch.tensor([True, False, True], dtype=torch.bool).numpy()
    input2 = torch.tensor([False, True, True], dtype=torch.bool).numpy()
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = torch.randint(0, 10, (2, 2), dtype=torch.int64).numpy()
    input2 = torch.randint(0, 10, (2, 2), dtype=torch.int64).numpy()
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = np.array([1,2,3], dtype=np.int16)
    input2 = np.array([3,2,1], dtype=np.int16)
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

list_of_inputs = bitwise_and_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('bitwise_and', list_of_inputs)
