
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def dirac__inputs():
    list_of_inputs = []

    tensor = torch.zeros((4, 4, 3), dtype=torch.float32).numpy()
    offset = np.int64(1)
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "offset": offset}))

    tensor = torch.ones((8, 8, 3, 3), dtype=torch.float64).numpy()
    offset = np.int32(2)
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "offset": offset}))

    tensor = torch.randn((6, 6, 1, 1), dtype=torch.float32).numpy()
    offset = np.int64(3)
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "offset": offset}))

    tensor = torch.zeros((2, 2, 5, 5, 5), dtype=torch.float32).numpy()
    offset = np.int32(1)
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "offset": offset}))

    tensor = torch.ones((12, 12, 1), dtype=torch.float16).numpy()
    offset = np.int64(6)
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "offset": offset}))

    tensor = torch.zeros((16, 16, 3, 1), dtype=torch.float32).numpy()
    offset = np.int32(8)
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "offset": offset}))

    tensor = torch.randn((10, 10, 1, 3), dtype=torch.float64).numpy()
    offset = np.int64(5)
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "offset": offset}))

    tensor = torch.zeros((2, 2, 1, 1), dtype=torch.float32).numpy()
    offset = np.int32(2)
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "offset": offset}))

    tensor = torch.ones((32, 32, 7, 7), dtype=torch.float32).numpy()
    offset = np.int64(16)
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "offset": offset}))

    tensor = torch.zeros((1, 1, 9), dtype=torch.float32).numpy()
    offset = np.int8(1)
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "offset": offset}))

    return list_of_inputs

generated_inputs["torch.nn.init.dirac_"] = dirac__inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.init.dirac_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.init.dirac_'.")


check_valid('torch.nn.init.dirac_', generated_inputs['torch.nn.init.dirac_'], lib="torch", suffix=0)
