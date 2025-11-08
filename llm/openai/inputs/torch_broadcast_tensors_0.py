
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def broadcast_tensors_inputs():
    list_of_inputs = []
    
    tensors = np.array([1.0, -2.5, 3.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"tensors": tensors}))
    
    tensors = np.array([7], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"tensors": tensors}))
    
    tensors = np.array([10.0, -20.0, 30.0, 0.0, 5.5], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"tensors": tensors}))
    
    tensors = np.array([True, False, True, False], dtype=bool)
    list_of_inputs.append(copy.deepcopy({"tensors": tensors}))
    
    tensors = np.array([1+2j, -1-0.5j, 0+0j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"tensors": tensors}))
    
    tensors = np.array([0, 1, 2, 3, 4, 255], dtype=np.uint8)
    list_of_inputs.append(copy.deepcopy({"tensors": tensors}))
    
    tensors = np.array([-128, -64, 0, 64], dtype=np.int8)
    list_of_inputs.append(copy.deepcopy({"tensors": tensors}))
    
    tensors = np.array([0.125, -0.5], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"tensors": tensors}))
    
    tensors = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"tensors": tensors}))
    
    tensors = np.array([3-4j, -2+1j], dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"tensors": tensors}))
    
    tensors = np.arange(-5, 5, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"tensors": tensors}))
    
    tensors = np.array([np.inf, -np.inf, np.nan, 0.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"tensors": tensors}))
    
    return list_of_inputs

generated_inputs["torch.broadcast_tensors"] = broadcast_tensors_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.broadcast_tensors' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.broadcast_tensors'.")


check_valid('torch.broadcast_tensors', generated_inputs['torch.broadcast_tensors'], lib="torch", suffix=0)
