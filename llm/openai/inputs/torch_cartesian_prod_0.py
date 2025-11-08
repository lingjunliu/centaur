
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def cartesian_prod_inputs():
    list_of_inputs = []

    tlist = np.array([1, 2, 3], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"tensors": tlist}))

    tlist = np.array([-5, 0, 5, 10], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"tensors": tlist}))

    tlist = np.array([0.0, -1.5, 2.3], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"tensors": tlist}))

    tlist = np.array([True, False, True], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"tensors": tlist}))

    tlist = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"tensors": tlist}))

    tlist = np.array([-(2**60), 0, 2**60], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"tensors": tlist}))

    tlist = np.array([-32768, 0, 32767], dtype=np.int16)
    list_of_inputs.append(copy.deepcopy({"tensors": tlist}))

    tlist = np.array([-1.5, 0.0, 2.5], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"tensors": tlist}))

    tlist = np.array([0, 128, 255], dtype=np.uint8)
    list_of_inputs.append(copy.deepcopy({"tensors": tlist}))

    tlist = np.array([-128, 0, 127], dtype=np.int8)
    list_of_inputs.append(copy.deepcopy({"tensors": tlist}))

    tlist = np.array([-1.0, 0.0, 1.0, 3.14159], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"tensors": tlist}))

    tlist = np.array([7], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"tensors": tlist}))

    return list_of_inputs

generated_inputs["torch.cartesian_prod"] = cartesian_prod_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.cartesian_prod' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.cartesian_prod'.")


check_valid('torch.cartesian_prod', generated_inputs['torch.cartesian_prod'], lib="torch", suffix=0)
