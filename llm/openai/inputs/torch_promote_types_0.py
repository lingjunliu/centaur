
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def promote_types_inputs():
    list_of_inputs = []

    type1 = np.dtype('int8'); type2 = np.dtype('int16')
    list_of_inputs.append(copy.deepcopy({"type1": type1, "type2": type2}))

    type1 = np.dtype('int32'); type2 = np.dtype('float32')
    list_of_inputs.append(copy.deepcopy({"type1": type1, "type2": type2}))

    type1 = np.dtype('uint8'); type2 = np.dtype('int16')
    list_of_inputs.append(copy.deepcopy({"type1": type1, "type2": type2}))

    type1 = np.dtype('bool'); type2 = np.dtype('int64')
    list_of_inputs.append(copy.deepcopy({"type1": type1, "type2": type2}))

    type1 = np.dtype('float16'); type2 = np.dtype('float64')
    list_of_inputs.append(copy.deepcopy({"type1": type1, "type2": type2}))

    type1 = np.dtype('float32'); type2 = np.dtype('complex64')
    list_of_inputs.append(copy.deepcopy({"type1": type1, "type2": type2}))

    type1 = np.dtype('int64'); type2 = np.dtype('complex128')
    list_of_inputs.append(copy.deepcopy({"type1": type1, "type2": type2}))

    type1 = np.dtype('uint8'); type2 = np.dtype('float32')
    list_of_inputs.append(copy.deepcopy({"type1": type1, "type2": type2}))

    type1 = np.dtype('int32'); type2 = np.dtype('int32')
    list_of_inputs.append(copy.deepcopy({"type1": type1, "type2": type2}))

    type1 = np.dtype('float64'); type2 = np.dtype('complex64')
    list_of_inputs.append(copy.deepcopy({"type1": type1, "type2": type2}))

    type1 = np.dtype('bool'); type2 = np.dtype('float16')
    list_of_inputs.append(copy.deepcopy({"type1": type1, "type2": type2}))

    type1 = np.dtype('complex64'); type2 = np.dtype('complex128')
    list_of_inputs.append(copy.deepcopy({"type1": type1, "type2": type2}))

    type1 = np.dtype('int8'); type2 = np.dtype('bool')
    list_of_inputs.append(copy.deepcopy({"type1": type1, "type2": type2}))

    type1 = np.dtype('uint8'); type2 = np.dtype('complex128')
    list_of_inputs.append(copy.deepcopy({"type1": type1, "type2": type2}))

    return list_of_inputs

generated_inputs["torch.promote_types"] = promote_types_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.promote_types' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.promote_types'.")


check_valid('torch.promote_types', generated_inputs['torch.promote_types'], lib="torch", suffix=0)
