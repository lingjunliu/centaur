
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def ShortStorage_inputs():
    list_of_inputs = []

    # Input 1: Empty numpy array
    input1 = {'data': list(np.array([], dtype=np.int16))}
    list_of_inputs.append(input1)

    # Input 2: 1D numpy array with positive shorts
    input2 = {'data': list(np.array([1, 2, 3, 4, 5], dtype=np.int16))}
    list_of_inputs.append(input2)

    # Input 3: 1D numpy array with negative shorts
    input3 = {'data': list(np.array([-1, -2, -3, -4, -5], dtype=np.int16))}
    list_of_inputs.append(input3)

    # Input 4: 2D numpy array with shorts
    input4 = {'data': list(np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int16).flatten())}
    list_of_inputs.append(input4)

    # Input 5: 1D numpy array with mixed positive and negative shorts
    input5 = {'data': list(np.array([-1, 2, -3, 4, -5], dtype=np.int16))}
    list_of_inputs.append(input5)
    
    return list_of_inputs

generated_inputs["torch.ShortStorage_5"] = ShortStorage_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.ShortStorage_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.ShortStorage_5'.")

check_valid('torch.ShortStorage', generated_inputs['torch.ShortStorage_5'], lib="torch")
