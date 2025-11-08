
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_dequantize_inputs():
    list_of_inputs = []

    tensor = np.array([0, 127, 255], dtype=np.uint8)
    list_of_inputs.append(copy.deepcopy({"tensor": tensor}))

    tensor = np.array([-128, -1, 0, 1, 127], dtype=np.int8)
    list_of_inputs.append(copy.deepcopy({"tensor": tensor}))

    tensor = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
    list_of_inputs.append(copy.deepcopy({"tensor": tensor}))

    tensor = (np.arange(24, dtype=np.int8) - 12).reshape(2, 3, 4)
    list_of_inputs.append(copy.deepcopy({"tensor": tensor}))

    tensor = np.array(128, dtype=np.uint8)
    list_of_inputs.append(copy.deepcopy({"tensor": tensor}))

    tensor = np.arange(100, dtype=np.uint8).reshape(10, 10)[::3, ::2]
    list_of_inputs.append(copy.deepcopy({"tensor": tensor}))

    tensor = np.asfortranarray(np.arange(12, dtype=np.int8).reshape(3, 4))
    list_of_inputs.append(copy.deepcopy({"tensor": tensor}))

    tensor = np.arange(12, dtype=np.uint8).reshape(3, 4).T
    list_of_inputs.append(copy.deepcopy({"tensor": tensor}))

    tensor = np.empty((0,), dtype=np.uint8)
    list_of_inputs.append(copy.deepcopy({"tensor": tensor}))

    tensor = np.zeros((1, 0, 2), dtype=np.int8)
    list_of_inputs.append(copy.deepcopy({"tensor": tensor}))

    tensor = np.arange(32, dtype=np.int8).reshape(2, 2, 2, 2, 2)
    list_of_inputs.append(copy.deepcopy({"tensor": tensor}))

    tensor = np.array([[1000, -1000, 0],
                       [2147483647, -2147483648, 12345]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"tensor": tensor}))

    return list_of_inputs

generated_inputs["torch.dequantize_1"] = torch_dequantize_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.dequantize_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.dequantize_1'.")


check_valid('torch.dequantize', generated_inputs['torch.dequantize_1'], lib="torch", suffix=1)
