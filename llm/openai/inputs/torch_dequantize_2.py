
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def dequantize_2_inputs():
    list_of_inputs = []

    tensors = np.array([0, 127, -128, 64, -64], dtype=np.int8)
    list_of_inputs.append(copy.deepcopy({"tensors": tensors}))

    tensors = np.array([[0, 255], [128, 64]], dtype=np.uint8)
    list_of_inputs.append(copy.deepcopy({"tensors": tensors}))

    tensors = np.array([[[1, -1], [2, -2]], [[3, -3], [4, -4]]], dtype=np.int16)
    list_of_inputs.append(copy.deepcopy({"tensors": tensors}))

    tensors = np.array([-2147483648, -1, 0, 1, 2147483647], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"tensors": tensors}))

    tensors = np.random.randint(-128, 128, size=(2, 3, 4, 5), dtype=np.int8)
    list_of_inputs.append(copy.deepcopy({"tensors": tensors}))

    tensors = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
    list_of_inputs.append(copy.deepcopy({"tensors": tensors}))

    base = np.arange(-60, 60, dtype=np.int8)
    tensors = base[::3]
    list_of_inputs.append(copy.deepcopy({"tensors": tensors}))

    tensors = np.array(-7, dtype=np.int8)
    list_of_inputs.append(copy.deepcopy({"tensors": tensors}))

    tensors = np.array([], dtype=np.int8)
    list_of_inputs.append(copy.deepcopy({"tensors": tensors}))

    tensors = np.empty((2, 0, 3), dtype=np.int8)
    list_of_inputs.append(copy.deepcopy({"tensors": tensors}))

    tensors = np.arange(-12, 0, dtype=np.int8).reshape(3, 4).T
    list_of_inputs.append(copy.deepcopy({"tensors": tensors}))

    tensors = np.linspace(-1.0, 1.0, num=12, dtype=np.float32).reshape(3, 4)
    list_of_inputs.append(copy.deepcopy({"tensors": tensors}))

    return list_of_inputs

generated_inputs["torch.dequantize_2"] = dequantize_2_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.dequantize_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.dequantize_2'.")


check_valid('torch.dequantize', generated_inputs['torch.dequantize_2'], lib="torch", suffix=2)
