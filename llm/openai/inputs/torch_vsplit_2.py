
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def vsplit_inputs():
    list_of_inputs = []

    # Input 1
    input = torch.arange(16.0).reshape(4, 4).numpy()
    indices_or_sections = [2]
    input_dict = {"input": input, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input = torch.arange(15).reshape(5, 3).numpy()
    indices_or_sections = [0, 2, 5]
    input_dict = {"input": input, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input = torch.randn(6, 2, 2, dtype=torch.float32).numpy()
    indices_or_sections = [2, 4]
    input_dict = {"input": input, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input = torch.randint(0, 2, (3, 2, 2, 2), dtype=torch.int8).bool().numpy()
    indices_or_sections = [1, 1, 3]
    input_dict = {"input": input, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input = torch.randn(7, 1, dtype=torch.complex64).numpy()
    indices_or_sections = [3, 6, 10]
    input_dict = {"input": input, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input = torch.arange(25).reshape(5, 5).numpy()
    indices_or_sections = [-5, -2]
    input_dict = {"input": input, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input = torch.empty((0, 4), dtype=torch.float32).numpy()
    indices_or_sections = [0, 2]
    input_dict = {"input": input, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input = torch.randint(-100, 100, (4, 3, 5), dtype=torch.int16).numpy()
    indices_or_sections = [1, 2, 3, 4]
    input_dict = {"input": input, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input = np.arange(50, dtype=np.float64).reshape(10, 5)[::2]
    indices_or_sections = [1, 4]
    input_dict = {"input": input, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input = torch.zeros((8, 1, 1, 1, 1), dtype=torch.int8).numpy()
    indices_or_sections = [2, 2, 5, 7, 8]
    input_dict = {"input": input, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    input = torch.randn(5, 0, dtype=torch.float64).numpy()
    indices_or_sections = [1, 3, 5]
    input_dict = {"input": input, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    input = torch.randn(9, 4, dtype=torch.float16).numpy()
    indices_or_sections = [3, 3, 6, 9, 12]
    input_dict = {"input": input, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.vsplit_2"] = vsplit_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.vsplit_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.vsplit_2'.")


check_valid('torch.vsplit', generated_inputs['torch.vsplit_2'], lib="torch", suffix=2)
