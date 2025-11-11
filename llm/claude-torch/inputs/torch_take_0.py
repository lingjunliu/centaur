
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import copy
import numpy as np

def take_inputs():
    list_of_inputs = []
    
    input_tensor = np.array([[4, 3, 5], [6, 7, 8]])
    index = np.array([0, 2, 5], dtype=np.int64)
    input_dict = {"input": input_tensor, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([10, 20, 30, 40, 50])
    index = np.array([2], dtype=np.int64)
    input_dict = {"input": input_tensor, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    index = np.array([0, 3, 7], dtype=np.int64)
    input_dict = {"input": input_tensor, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.arange(100)
    index = np.array([5, 10, 50, 99, 0], dtype=np.int64)
    input_dict = {"input": input_tensor, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]])
    index = np.array([[0, 1], [2, 5]], dtype=np.int64)
    input_dict = {"input": input_tensor, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([42])
    index = np.array([0], dtype=np.int64)
    input_dict = {"input": input_tensor, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.ones((2, 2, 2, 2))
    index = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    input_dict = {"input": input_tensor, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
    index = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    input_dict = {"input": input_tensor, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[-5, -10, -15], [-20, -25, -30]])
    index = np.array([1, 4], dtype=np.int64)
    input_dict = {"input": input_tensor, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.arange(24).reshape(4, 6)
    index = np.array([23, 20, 15, 10, 5, 0], dtype=np.int64)
    input_dict = {"input": input_tensor, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.take"] = take_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.take' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.take'.")


check_valid('torch.take', generated_inputs['torch.take'], lib="torch", suffix=0)
