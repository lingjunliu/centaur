
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def pixel_unshuffle_inputs():
    list_of_inputs = []

    # Input 1: Simple 4D tensor
    input1 = torch.randn(1, 4, 8, 8).numpy()
    downscale_factor1 = 2
    input_dict1 = {"input": input1, "downscale_factor": downscale_factor1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Different channel size
    input2 = torch.randn(1, 16, 8, 8).numpy()
    downscale_factor2 = 4
    input_dict2 = {"input": input2, "downscale_factor": downscale_factor2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Different batch size
    input3 = torch.randn(2, 4, 8, 8).numpy()
    downscale_factor3 = 2
    input_dict3 = {"input": input3, "downscale_factor": downscale_factor3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Larger image size
    input4 = torch.randn(1, 4, 16, 16).numpy()
    downscale_factor4 = 2
    input_dict4 = {"input": input4, "downscale_factor": downscale_factor4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5: Different downscale factor and channel size
    input5 = torch.randn(1, 9, 9, 9).numpy()
    downscale_factor5 = 3
    input_dict5 = {"input": input5, "downscale_factor": downscale_factor5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Non-square image
    input6 = torch.randn(1, 4, 8, 16).numpy()
    downscale_factor6 = 2
    input_dict6 = {"input": input6, "downscale_factor": downscale_factor6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: Small image size
    input7 = torch.randn(1, 4, 4, 4).numpy()
    downscale_factor7 = 2
    input_dict7 = {"input": input7, "downscale_factor": downscale_factor7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: Batch size > 1 and different image size
    input8 = torch.randn(3, 4, 12, 12).numpy()
    downscale_factor8 = 2
    input_dict8 = {"input": input8, "downscale_factor": downscale_factor8}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    # Input 9: Ensure it works with downscale factor of 1
    input9 = torch.randn(1, 4, 8, 8).numpy()
    downscale_factor9 = 1
    input_dict9 = {"input": input9, "downscale_factor": downscale_factor9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10:  Large values
    input10 = torch.randint(1,100,(1, 4, 8, 8)).numpy()
    downscale_factor10 = 2
    input_dict10 = {"input": input10, "downscale_factor": downscale_factor10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.pixel_unshuffle"] = pixel_unshuffle_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.pixel_unshuffle' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.pixel_unshuffle'.")

check_valid('torch.nn.functional.pixel_unshuffle', generated_inputs['torch.nn.functional.pixel_unshuffle'], lib="torch", suffix=0)
