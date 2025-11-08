
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def init_zeros__inputs():
    list_of_inputs = []

    tensor = np.array([1.0, -2.5, 3.3, 0.0, 9.9], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"tensor": tensor}))

    tensor = np.array([[1.5, -0.5, 3.0],
                       [4.2, 5.5, -6.1]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"tensor": tensor}))

    tensor = np.array(7, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"tensor": tensor}))

    tensor = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"tensor": tensor}))

    tensor = np.empty((0, 4), dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"tensor": tensor}))

    tensor = np.array([[[1, -2, 3],
                        [4, 5, -6]]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"tensor": tensor}))

    tensor = np.array([[255, 0, 128],
                       [64, 32, 16]], dtype=np.uint8)
    list_of_inputs.append(copy.deepcopy({"tensor": tensor}))

    tensor = np.array([[True, False, True],
                       [False, False, True],
                       [True, True, False]], dtype=bool)
    list_of_inputs.append(copy.deepcopy({"tensor": tensor}))

    tensor = np.arange(16, dtype=np.float16).reshape(2, 2, 2, 2)
    list_of_inputs.append(copy.deepcopy({"tensor": tensor}))

    tensor = np.array([[1 + 2j, -3 + 0.5j],
                       [0 - 1j, 2 + 0j]], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"tensor": tensor}))

    tensor = np.array([0 + 0j, 1 - 1j, -2 + 2j], dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"tensor": tensor}))

    base = np.arange(12, dtype=np.float32).reshape(3, 4)
    tensor = base.T
    list_of_inputs.append(copy.deepcopy({"tensor": tensor}))

    return list_of_inputs

generated_inputs["torch.nn.init.zeros_"] = init_zeros__inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.init.zeros_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.init.zeros_'.")


check_valid('torch.nn.init.zeros_', generated_inputs['torch.nn.init.zeros_'], lib="torch", suffix=0)
