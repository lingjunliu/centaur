
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def is_inference_inputs():
    list_of_inputs = []
    
    # Input 1
    input_arr = np.array([1.0])
    input_dict = {"input": torch.tensor(input_arr)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2
    input_arr = np.array([1, 2, 3])
    input_dict = {"input": torch.tensor(input_arr)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    input_dict = {"input": torch.tensor(input_arr)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4
    input_arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    input_dict = {"input": torch.tensor(input_arr)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    input_arr = np.array([1])
    input_dict = {"input": torch.tensor(input_arr)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_arr = np.array([1, 2, 3], dtype=np.int32)
    input_dict = {"input": torch.tensor(input_arr)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    input_dict = {"input": torch.tensor(input_arr)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    input_arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64)
    input_dict = {"input": torch.tensor(input_arr)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    input_arr = np.array([-1, -2, -3])
    input_dict = {"input": torch.tensor(input_arr)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_arr = np.array([0.0])
    input_dict = {"input":  torch.tensor(input_arr)}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.is_inference"] = is_inference_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.is_inference' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.is_inference'.")

check_valid('torch.is_inference', generated_inputs['torch.is_inference'], lib="torch", suffix=0)
