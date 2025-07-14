
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def adaptive_max_pool2d_inputs():
    list_of_inputs = []

    # Input 1: Basic case with square output size
    input_tensor = torch.randn(1, 3, 32, 32).numpy()
    output_size = (16, 16)
    return_indices = False
    input_dict = {"output_size": output_size, "return_indices": return_indices, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Rectangular output size
    input_tensor = torch.randn(1, 3, 64, 128).numpy()
    output_size = (32, 64)
    return_indices = True
    input_dict = {"output_size": output_size, "return_indices": return_indices, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Batch size > 1
    input_tensor = torch.randn(4, 3, 32, 32).numpy()
    output_size = (8, 8)
    return_indices = False
    input_dict = {"output_size": output_size, "return_indices": return_indices, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different number of input channels
    input_tensor = torch.randn(1, 64, 32, 32).numpy()
    output_size = (16, 16)
    return_indices = True
    input_dict = {"output_size": output_size, "return_indices": return_indices, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Smaller input size
    input_tensor = torch.randn(1, 3, 16, 16).numpy()
    output_size = (8, 8)
    return_indices = False
    input_dict = {"output_size": output_size, "return_indices": return_indices, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6:  output_size with None, remove None to avoid TypeError
    input_tensor = torch.randn(1, 3, 16, 16).numpy()
    output_size = (16, 8)
    return_indices = False
    input_dict = {"output_size": output_size, "return_indices": return_indices, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7:  output_size with different input shape, remove None to avoid TypeError
    input_tensor = torch.randn(2, 5, 32, 64).numpy()
    output_size = (16, 32)
    return_indices = True
    input_dict = {"output_size": output_size, "return_indices": return_indices, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: single channel input
    input_tensor = torch.randn(1, 1, 32, 32).numpy()
    output_size = (8, 8)
    return_indices = False
    input_dict = {"output_size": output_size, "return_indices": return_indices, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: C,Hin,Win input
    input_tensor = torch.randn(3, 32, 32).numpy()
    output_size = (8, 8)
    return_indices = True
    input_dict = {"output_size": output_size, "return_indices": return_indices, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Different input dimensions.
    input_tensor = torch.randn(2, 3, 128, 256).numpy()
    output_size = (64, 128)
    return_indices = False
    input_dict = {"output_size": output_size, "return_indices": return_indices, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: square image with int output size
    input_tensor = torch.randn(1, 3, 16, 16).numpy()
    output_size = (7,7) #make it a tuple
    return_indices = False
    input_dict = {"output_size": output_size, "return_indices": return_indices, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.AdaptiveMaxPool2d_2"] = adaptive_max_pool2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.AdaptiveMaxPool2d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.AdaptiveMaxPool2d_2'.")

check_valid('torch.nn.AdaptiveMaxPool2d', generated_inputs['torch.nn.AdaptiveMaxPool2d_2'], lib="torch", suffix=2)
