
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def ones__inputs():
    list_of_inputs = []
    
    tensor = np.array([0.0, -3.14, 2.718], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"tensor": tensor}))
    
    tensor = np.zeros((2, 3), dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"tensor": tensor}))
    
    tensor = np.random.randn(2, 1, 4).astype(np.float16)
    list_of_inputs.append(copy.deepcopy({"tensor": tensor}))
    
    tensor = np.array(5.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"tensor": tensor}))
    
    tensor = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"tensor": tensor}))
    
    tensor = np.empty((2, 0, 3), dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"tensor": tensor}))
    
    tensor = np.array([[-1, 0, 1],
                       [2, -3, 4],
                       [5, 6, -7]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"tensor": tensor}))
    
    tensor = np.array([[True, False],
                       [False, True]], dtype=bool)
    list_of_inputs.append(copy.deepcopy({"tensor": tensor}))
    
    tensor = np.array([1+2j, -3+0.5j, -1j, 2+0j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"tensor": tensor}))
    
    base = np.arange(24, dtype=np.float32).reshape(4, 6)
    tensor = base[::2, ::3]
    list_of_inputs.append(copy.deepcopy({"tensor": tensor}))
    
    tensor = np.asfortranarray(np.arange(12., dtype=np.float64).reshape(3, 4))
    list_of_inputs.append(copy.deepcopy({"tensor": tensor}))
    
    tensor = np.arange(12, dtype=np.int32).reshape(3, 4).T
    list_of_inputs.append(copy.deepcopy({"tensor": tensor}))
    
    tensor = (np.ones((1, 2, 3, 4), dtype=np.float32) * -5)
    list_of_inputs.append(copy.deepcopy({"tensor": tensor}))
    
    tensor = np.array([[1+0j, -2-3j],
                       [4.5+6.7j, 0-1j]], dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"tensor": tensor}))
    
    return list_of_inputs

generated_inputs["torch.nn.init.ones_"] = ones__inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.init.ones_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.init.ones_'.")


check_valid('torch.nn.init.ones_', generated_inputs['torch.nn.init.ones_'], lib="torch", suffix=0)
