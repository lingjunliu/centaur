
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def unbind_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D tensor
    input_tensor = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    dim = 0
    input_dict = {"input": input_tensor, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3D tensor, different dimension
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    dim = 1
    input_dict = {"input": input_tensor, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D tensor
    input_tensor = np.array([1, 2, 3, 4, 5])
    dim = 0
    input_dict = {"input": input_tensor, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Float tensor
    input_tensor = np.array([[1.1, 2.2], [3.3, 4.4]])
    dim = 0
    input_dict = {"input": input_tensor, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative values and different dim
    input_tensor = np.array([[-1, 2, -3], [4, -5, 6]])
    dim = 1
    input_dict = {"input": input_tensor, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D tensor
    input_tensor = np.random.rand(2, 3, 4, 5)
    dim = 2
    input_dict = {"input": input_tensor, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Tensor with a dimension of size 1
    input_tensor = np.array([[[1], [2], [3]]])
    dim = 0
    input_dict = {"input": input_tensor, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.unbind"] = unbind_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.unbind' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.unbind'.")

check_valid('torch.unbind', generated_inputs['torch.unbind'], lib="torch")
