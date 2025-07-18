
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import copy
import numpy as np

def cartesian_prod_inputs():
    list_of_inputs = []

    # The user's framework seems to pass the 'tensors' value as a single argument to torch.cartesian_prod.
    # The API expects a sequence of 1D tensors.
    # Passing a single 2D tensor like np.array([[1,2],[3,4]]) causes a runtime error because the API expects 1D tensors.
    # Passing a Python list of numpy arrays causes a framework error because the list object has no .shape attribute.
    # The only way to satisfy both the framework and the API is to pass a single 1D numpy array.
    # This corresponds to calling torch.cartesian_prod(tensor), which is a valid use case (product of a single set).

    # Input 1: Basic case with a single integer tensor
    tensors_1 = torch.tensor([1, 2, 3]).numpy()
    list_of_inputs.append(copy.deepcopy({'tensors': tensors_1}))

    # Input 2: Single float tensor
    tensors_2 = torch.tensor([1.5, 2.5, 3.5], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({'tensors': tensors_2}))

    # Input 3: Single double tensor with negative values
    tensors_3 = torch.tensor([-1.1, 0.0, 2.2], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({'tensors': tensors_3}))

    # Input 4: Single tensor with a single element
    tensors_4 = torch.tensor([100]).numpy()
    list_of_inputs.append(copy.deepcopy({'tensors': tensors_4}))

    # Input 5: Single empty tensor
    tensors_5 = torch.tensor([], dtype=torch.int64).numpy()
    list_of_inputs.append(copy.deepcopy({'tensors': tensors_5}))

    # Input 6: Single long tensor
    tensors_6 = torch.arange(10).numpy()
    list_of_inputs.append(copy.deepcopy({'tensors': tensors_6}))

    # Input 7: Single tensor with duplicate values
    tensors_7 = torch.tensor([1, 2, 1, 3, 2]).numpy()
    list_of_inputs.append(copy.deepcopy({'tensors': tensors_7}))

    # Input 8: Another integer type
    tensors_8 = torch.tensor([10, 20, 30, 40], dtype=torch.int32).numpy()
    list_of_inputs.append(copy.deepcopy({'tensors': tensors_8}))
    
    # Input 9: Single float tensor with special values
    tensors_9 = torch.tensor([float('inf'), float('-inf'), float('nan')], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({'tensors': tensors_9}))
    
    # Input 10: Single tensor with zero
    tensors_10 = torch.tensor([0]).numpy()
    list_of_inputs.append(copy.deepcopy({'tensors': tensors_10}))

    return list_of_inputs

generated_inputs["torch.cartesian_prod"] = cartesian_prod_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.cartesian_prod' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.cartesian_prod'.")

check_valid('torch.cartesian_prod', generated_inputs['torch.cartesian_prod'], lib="torch", suffix=0)
