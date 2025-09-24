
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def unsafe_split_with_sizes_inputs():
    list_of_inputs = []

    # Test case 1: Basic 1D tensor splitting
    input1 = torch.arange(10).numpy()
    split_sizes1 = [2, 3, 5]
    dim1 = 0
    input_dict1 = {"input": input1, "split_sizes": split_sizes1, "dim": dim1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Test case 2: 2D tensor splitting along dim=0
    input2 = torch.randn(5, 4).numpy()
    split_sizes2 = [1, 2, 2]
    dim2 = 0
    input_dict2 = {"input": input2, "split_sizes": split_sizes2, "dim": dim2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Test case 3: 2D tensor splitting along dim=1
    input3 = torch.randn(3, 6).numpy()
    split_sizes3 = [2, 1, 3]
    dim3 = 1
    input_dict3 = {"input": input3, "split_sizes": split_sizes3, "dim": dim3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Test case 4: 3D tensor splitting
    input4 = torch.randn(2, 3, 4).numpy()
    split_sizes4 = [1, 1]
    dim4 = 0
    input_dict4 = {"input": input4, "split_sizes": split_sizes4, "dim": dim4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Test case 5: Splitting a tensor with float data type
    input5 = torch.randn(5).float().numpy()
    split_sizes5 = [2, 3]
    dim5 = 0
    input_dict5 = {"input": input5, "split_sizes": split_sizes5, "dim": dim5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Test case 6: Splitting a tensor with integer data type
    input6 = torch.randint(0, 10, (7,)).int().numpy()
    split_sizes6 = [2, 2, 3]
    dim6 = 0
    input_dict6 = {"input": input6, "split_sizes": split_sizes6, "dim": dim6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Test case 7: Splitting with different split_sizes types (list of tensors)
    input7 = torch.arange(10).numpy()
    split_sizes7 = [torch.tensor(2), torch.tensor(3), torch.tensor(5)]
    dim7 = 0
    input_dict7 = {"input": input7, "split_sizes": split_sizes7, "dim": dim7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Test case 8: 4D tensor
    input8 = torch.randn(2, 3, 4, 5).numpy()
    split_sizes8 = [1, 1]
    dim8 = 0
    input_dict8 = {"input": input8, "split_sizes": split_sizes8, "dim": dim8}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    return list_of_inputs

generated_inputs["torch.unsafe_split_with_sizes"] = unsafe_split_with_sizes_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.unsafe_split_with_sizes' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.unsafe_split_with_sizes'.")

check_valid('torch.unsafe_split_with_sizes', generated_inputs['torch.unsafe_split_with_sizes'], lib="torch")
