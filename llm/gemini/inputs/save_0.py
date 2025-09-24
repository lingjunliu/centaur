
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import io
import pickle
import copy

def torch_save_inputs():
    list_of_inputs = []

    # Input 1: Saving a simple numpy array to a file
    obj = np.array([1, 2, 3, 4, 5])
    f = "test_tensor1.pt"
    pickle_module = None
    pickle_protocol = 2
    input_dict = {
        "obj": obj,
        "f": f,
        "pickle_module": pickle_module,
        "pickle_protocol": pickle_protocol
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Saving a multi-dimensional numpy array to a BytesIO buffer
    obj = np.random.rand(2, 3, 4)
    f = io.BytesIO()
    pickle_module = None
    pickle_protocol = 4
    input_dict = {
        "obj": obj,
        "f": f,
        "pickle_module": pickle_module,
        "pickle_protocol": pickle_protocol
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Saving a numpy array with a different pickle module
    obj = np.array([[1.0, 2.0], [3.0, 4.0]])
    f = "test_tensor2.pt"
    pickle_module = None
    pickle_protocol = 3
    input_dict = {
        "obj": obj,
        "f": f,
        "pickle_module": pickle_module,
        "pickle_protocol": pickle_protocol
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Saving a numpy array containing negative values
    obj = np.array([-1, -2, 0, 1, 2])
    f = "test_tensor3.pt"
    pickle_module = None
    pickle_protocol = 2
    input_dict = {
        "obj": obj,
        "f": f,
        "pickle_module": pickle_module,
        "pickle_protocol": pickle_protocol
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Saving a zero dimensional numpy array.
    obj = np.array(5)
    f = "test_tensor4.pt"
    pickle_module = None
    pickle_protocol = 2
    input_dict = {
        "obj": obj,
        "f": f,
        "pickle_module": pickle_module,
        "pickle_protocol": pickle_protocol
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.save"] = torch_save_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.save' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.save'.")

check_valid('torch.save', generated_inputs['torch.save'], lib="torch")
