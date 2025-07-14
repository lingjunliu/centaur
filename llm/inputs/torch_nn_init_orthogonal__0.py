
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def orthogonal_inputs():
    list_of_inputs = []

    # Input 1: 2D tensor
    tensor = np.zeros((10, 10), dtype=np.float32)
    gain = 1.0
    input_dict = {"tensor": tensor, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3D tensor
    tensor = np.zeros((5, 5, 5), dtype=np.float32)
    gain = 2.0
    input_dict = {"tensor": tensor, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Rectangular matrix
    tensor = np.zeros((10, 5), dtype=np.float32)
    gain = 0.5
    input_dict = {"tensor": tensor, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Small matrix
    tensor = np.zeros((2, 2), dtype=np.float32)
    gain = 1.5
    input_dict = {"tensor": tensor, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different gain
    tensor = np.zeros((7, 7), dtype=np.float32)
    gain = 0.75
    input_dict = {"tensor": tensor, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D Tensor
    tensor = np.zeros((2, 3, 4, 5), dtype=np.float32)
    gain = 1.0
    input_dict = {"tensor": tensor, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Larger gain value
    tensor = np.zeros((3, 3), dtype=np.float32)
    gain = 5.0
    input_dict = {"tensor": tensor, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Different dimensions, more rectangular
    tensor = np.zeros((3, 10), dtype=np.float32)
    gain = 1.0
    input_dict = {"tensor": tensor, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Float64 tensor
    tensor = np.zeros((5, 5), dtype=np.float64)
    gain = 1.0
    input_dict = {"tensor": tensor, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.init.orthogonal_"] = orthogonal_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.init.orthogonal_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.init.orthogonal_'.")

check_valid('torch.nn.init.orthogonal_', generated_inputs['torch.nn.init.orthogonal_'], lib="torch", suffix=0)
