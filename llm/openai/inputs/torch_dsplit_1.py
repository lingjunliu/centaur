
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def dsplit_inputs():
    list_of_inputs = []
    
    input = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    indices_or_sections = np.int64(2)
    list_of_inputs.append(copy.deepcopy({"input": input, "indices_or_sections": indices_or_sections}))
    
    input = np.full((1, 1, 6), fill_value=-3, dtype=np.int64)
    indices_or_sections = np.int32(3)
    list_of_inputs.append(copy.deepcopy({"input": input, "indices_or_sections": indices_or_sections}))
    
    input = np.random.randint(-50, 50, size=(4, 5, 8, 2)).astype(np.int16)
    indices_or_sections = np.int8(4)
    list_of_inputs.append(copy.deepcopy({"input": input, "indices_or_sections": indices_or_sections}))
    
    input = (np.arange(3 * 2 * 9).reshape(3, 2, 9) % 2 == 0)
    indices_or_sections = np.int16(3)
    list_of_inputs.append(copy.deepcopy({"input": input, "indices_or_sections": indices_or_sections}))
    
    input = np.arange(2 * 2 * 1 * 7, dtype=np.float64).reshape(2, 2, 1, 7)
    indices_or_sections = np.int8(1)
    list_of_inputs.append(copy.deepcopy({"input": input, "indices_or_sections": indices_or_sections}))
    
    real = np.random.randn(2, 3, 10).astype(np.float32)
    imag = np.random.randn(2, 3, 10).astype(np.float32)
    input = (real + 1j * imag).astype(np.complex64)
    indices_or_sections = np.int32(5)
    list_of_inputs.append(copy.deepcopy({"input": input, "indices_or_sections": indices_or_sections}))
    
    input = np.empty((1, 4, 0, 2), dtype=np.float32)
    indices_or_sections = np.int64(5)
    list_of_inputs.append(copy.deepcopy({"input": input, "indices_or_sections": indices_or_sections}))
    
    input = np.arange(1 * 2 * 15 * 1 * 1, dtype=np.uint8).reshape(1, 2, 15, 1, 1)
    indices_or_sections = np.int16(5)
    list_of_inputs.append(copy.deepcopy({"input": input, "indices_or_sections": indices_or_sections}))
    
    input = np.linspace(-1, 1, num=14).astype(np.float16).reshape(7, 1, 2)
    indices_or_sections = np.int8(2)
    list_of_inputs.append(copy.deepcopy({"input": input, "indices_or_sections": indices_or_sections}))
    
    input = np.linspace(-100, 100, num=3 * 3 * 18, dtype=np.float64).reshape(3, 3, 18)
    indices_or_sections = np.int32(9)
    list_of_inputs.append(copy.deepcopy({"input": input, "indices_or_sections": indices_or_sections}))
    
    input = np.random.randint(-128, 128, size=(2, 5, 4, 3, 2)).astype(np.int8)
    indices_or_sections = np.int16(2)
    list_of_inputs.append(copy.deepcopy({"input": input, "indices_or_sections": indices_or_sections}))
    
    input = np.arange(32, dtype=np.int32).reshape(1, 1, 32)
    indices_or_sections = np.int64(8)
    list_of_inputs.append(copy.deepcopy({"input": input, "indices_or_sections": indices_or_sections}))
    
    return list_of_inputs

generated_inputs["torch.dsplit_1"] = dsplit_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.dsplit_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.dsplit_1'.")


check_valid('torch.dsplit', generated_inputs['torch.dsplit_1'], lib="torch", suffix=1)
