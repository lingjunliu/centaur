
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def select_copy_inputs():
    list_of_inputs = []

    # Input 1: Basic test with positive index
    input1 = torch.randn(3, 4, 5).numpy()
    dim1 = 1
    index1 = 2
    out1 = torch.empty(3, 5).numpy()
    input_dict1 = {"input": input1, "dim": dim1, "index": index1, "out": out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Negative index and dimension
    input2 = torch.randint(0, 10, (2, 3, 4)).float().numpy()
    dim2 = -1
    index2 = -1
    out2 = torch.empty(2, 3).float().numpy()
    input_dict2 = {"input": input2, "dim": dim2, "index": index2, "out": out2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Float tensor
    input3 = torch.randn(2, 2, 2).numpy()
    dim3 = 0
    index3 = 1
    out3 = torch.empty(2, 2).numpy()
    input_dict3 = {"input": input3, "dim": dim3, "index": index3, "out": out3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    # Input 4: 1D tensor
    input4 = torch.arange(5).float().numpy()
    dim4 = 0
    index4 = 2
    out4 = np.array([0.]).reshape(1).astype(np.float32)
    input_dict4 = {"input": input4, "dim": dim4, "index": index4, "out": out4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    return list_of_inputs

generated_inputs["torch.select_copy"] = select_copy_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.select_copy' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.select_copy'.")

check_valid('torch.select_copy', generated_inputs['torch.select_copy'], lib="torch")
