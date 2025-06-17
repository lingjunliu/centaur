
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def pixel_shuffle_inputs():
    list_of_inputs = []

    # Input 1: Basic case with float tensor
    input1 = torch.randn(1, 9, 4, 4).numpy()
    upscale_factor1 = 3
    input_dict1 = {"input": input1, "upscale_factor": upscale_factor1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Different batch size and input channels, float tensor
    input2 = torch.randn(2, 4, 8, 8).numpy()
    upscale_factor2 = 2
    input_dict2 = {"input": input2, "upscale_factor": upscale_factor2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Integer tensor
    input3 = torch.randint(0, 10, (1, 4, 5, 5)).numpy()
    upscale_factor3 = 2
    input_dict3 = {"input": input3, "upscale_factor": upscale_factor3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 6: Large upscale factor
    input6 = torch.randn(1, 16, 2, 2).numpy()
    upscale_factor6 = 4
    input_dict6 = {"input": input6, "upscale_factor": upscale_factor6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Different input dimensions
    input7 = torch.randn(1, 25, 3, 3).numpy()
    upscale_factor7 = 5
    input_dict7 = {"input": input7, "upscale_factor": upscale_factor7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: Another valid case
    input8 = torch.randn(2, 64, 5, 5).numpy()
    upscale_factor8 = 8
    input_dict8 = {"input": input8, "upscale_factor": upscale_factor8}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    #Input 9: Different channel size and upscale factor
    input9 = torch.randn(1, 8, 4, 4).numpy()
    upscale_factor9 = 2
    input_dict9 = {"input": input9, "upscale_factor": upscale_factor9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    #Input 10: channel = upscale_factor^2
    input10 = torch.randn(1, 4, 4, 4).numpy()
    upscale_factor10 = 2
    input_dict10 = {"input": input10, "upscale_factor": upscale_factor10}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.pixel_shuffle"] = pixel_shuffle_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.pixel_shuffle' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.pixel_shuffle'.")

check_valid('torch.nn.functional.pixel_shuffle', generated_inputs['torch.nn.functional.pixel_shuffle'], lib="torch")
