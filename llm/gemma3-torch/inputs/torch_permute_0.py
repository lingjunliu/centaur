
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def permute_inputs():
    list_of_inputs = []

    input1 = torch.randn(2, 3, 5).numpy()
    dims1 = (2, 0, 1)
    input_dict1 = {"input": input1, "dims": dims1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(4, 2).numpy()
    dims2 = (1, 0)
    input_dict2 = {"input": input2, "dims": dims2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(3).numpy()
    dims3 = (0,)
    input_dict3 = {"input": input3, "dims": dims3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(2, 3, 4, 5).numpy()
    dims4 = (3, 1, 0, 2)
    input_dict4 = {"input": input4, "dims": dims4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(5, 4, 3, 2, 1).numpy()
    dims5 = (4, 3, 2, 1, 0)
    input_dict5 = {"input": input5, "dims": dims5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.randn(10, 10, 10).numpy()
    dims6 = (0, 1, 2)
    input_dict6 = {"input": input6, "dims": dims6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = torch.randn(2, 2, 2, 2).numpy()
    dims7 = (1, 0, 3, 2)
    input_dict7 = {"input": input7, "dims": dims7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = torch.randn(7, 8, 9).numpy()
    dims8 = (2, 1, 0)
    input_dict8 = {"input": input8, "dims": dims8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = torch.randn(6, 5).numpy()
    dims9 = (1, 0)
    input_dict9 = {"input": input9, "dims": dims9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = torch.randn(1, 2, 3, 4).numpy()
    dims10 = (3, 2, 1, 0)
    input_dict10 = {"input": input10, "dims": dims10}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    return list_of_inputs

generated_inputs["torch.permute"] = permute_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.permute' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.permute'.")


check_valid('torch.permute', generated_inputs['torch.permute'], lib="torch", suffix=0)
