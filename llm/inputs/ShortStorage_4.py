
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def ShortStorage_inputs():
    list_of_inputs = []

    # Input 1: Empty numpy array
    input1 = np.array([], dtype=np.int16)
    storage1 = torch.ShortStorage(torch.ShortTensor(input1).tolist())
    list_of_inputs.append({'storage': storage1})

    # Input 2: 1D numpy array with positive short integers
    input2 = np.array([1, 2, 3, 4, 5], dtype=np.int16)
    storage2 = torch.ShortStorage(torch.ShortTensor(input2).tolist())
    list_of_inputs.append({'storage': storage2})

    # Input 3: 1D numpy array with negative and positive short integers
    input3 = np.array([-1, 0, 1, -2, 2], dtype=np.int16)
    storage3 = torch.ShortStorage(torch.ShortTensor(input3).tolist())
    list_of_inputs.append({'storage': storage3})

    # Input 4: 2D numpy array
    input4 = np.array([[1, 2], [3, 4]], dtype=np.int16)
    storage4 = torch.ShortStorage(torch.ShortTensor(input4.flatten()).tolist())
    list_of_inputs.append({'storage': storage4})

    # Input 5: 3D numpy array
    input5 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int16)
    storage5 = torch.ShortStorage(torch.ShortTensor(input5.flatten()).tolist())
    list_of_inputs.append({'storage': storage5})
    
    # Input 6: numpy array with maximum and minimum short values
    input6 = np.array([np.iinfo(np.int16).max, np.iinfo(np.int16).min], dtype=np.int16)
    storage6 = torch.ShortStorage(torch.ShortTensor(input6).tolist())
    list_of_inputs.append({'storage': storage6})

    return list_of_inputs

generated_inputs["torch.ShortStorage_4"] = ShortStorage_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.ShortStorage_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.ShortStorage_4'.")

check_valid('torch.ShortStorage', generated_inputs['torch.ShortStorage_4'], lib="torch")
