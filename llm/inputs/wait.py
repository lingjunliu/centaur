
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy
from torch.jit import fork, wait

def jit_wait_inputs():
    list_of_inputs = []

    def dummy_function(x):
        return x * 2

    scripted_dummy = torch.jit.script(dummy_function)

    # Input 1: Float Tensor
    input_tensor1 = torch.randn(3, 4)
    future1 = fork(scripted_dummy, input_tensor1)
    input_dict1 = {"input": input_tensor1.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Int Tensor
    input_tensor2 = torch.randint(0, 10, (2, 2))
    future2 = fork(scripted_dummy, input_tensor2)
    input_dict2 = {"input": input_tensor2.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Different Shape Tensor
    input_tensor3 = torch.randn(1, 5, 5, 5)
    future3 = fork(scripted_dummy, input_tensor3)
    input_dict3 = {"input": input_tensor3.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    # Input 4: Scalar Tensor
    input_tensor4 = torch.tensor(5.0)
    future4 = fork(scripted_dummy, input_tensor4)
    input_dict4 = {"input": torch.tensor(5.0).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Bool Tensor
    input_tensor5 = torch.tensor([True, False, True])
    future5 = fork(scripted_dummy, input_tensor5)
    input_dict5 = {"input": torch.tensor([True, False, True]).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict5))
        
    return list_of_inputs

generated_inputs = jit_wait_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('wait', generated_inputs)
