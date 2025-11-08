
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def pixel_shuffle_inputs():
    list_of_inputs = []

    # Input 1
    input_arr = torch.arange(-16, 0, dtype=torch.float32).reshape(1, 4, 2, 2).numpy()
    upscale_factor = 2
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "upscale_factor": upscale_factor}))

    # Input 2
    input_arr = np.random.randn(2, 9, 3, 5).astype(np.float64)
    upscale_factor = np.int32(3)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "upscale_factor": upscale_factor}))

    # Input 3
    input_arr = torch.randn(4, 16, 1, 1, dtype=torch.float16).numpy()
    upscale_factor = 4
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "upscale_factor": upscale_factor}))

    # Input 4
    input_arr = np.arange(16, dtype=np.int32).reshape(1, 1, 4, 4)
    upscale_factor = 1
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "upscale_factor": upscale_factor}))

    # Input 5
    input_arr = np.random.uniform(-1.0, 1.0, size=(3, 25, 2, 3)).astype(np.float32)
    upscale_factor = np.int64(5)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "upscale_factor": upscale_factor}))

    # Input 6
    input_arr = np.random.randint(-100, 100, size=(2, 8, 7, 7)).astype(np.int16)
    upscale_factor = 2
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "upscale_factor": upscale_factor}))

    # Input 7
    input_arr = (np.ones((1, 36, 2, 2), dtype=np.float32) * 3.14).astype(np.float32)
    upscale_factor = 6
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "upscale_factor": upscale_factor}))

    # Input 8
    input_arr = np.linspace(-1, 1, num=5 * 4 * 8 * 6, dtype=np.float64).reshape(5, 4, 8, 6)
    upscale_factor = np.int32(2)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "upscale_factor": upscale_factor}))

    # Input 9
    input_arr = torch.randn(2, 12, 3, 4, dtype=torch.float32).numpy()
    upscale_factor = 2
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "upscale_factor": upscale_factor}))

    # Input 10
    input_arr = np.random.randn(1, 18, 2, 5).astype(np.float32)
    upscale_factor = np.int64(3)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "upscale_factor": upscale_factor}))

    # Input 11
    input_arr = np.arange(50, dtype=np.int64).reshape(1, 50, 1, 1)
    upscale_factor = 5
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "upscale_factor": upscale_factor}))

    # Input 12
    input_arr = torch.linspace(-5, 5, steps=1 * 32 * 3 * 3, dtype=torch.float32).reshape(1, 32, 3, 3).numpy()
    upscale_factor = 4
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "upscale_factor": upscale_factor}))

    return list_of_inputs

generated_inputs["torch.nn.functional.pixel_shuffle"] = pixel_shuffle_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.pixel_shuffle' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.pixel_shuffle'.")


check_valid('torch.nn.functional.pixel_shuffle', generated_inputs['torch.nn.functional.pixel_shuffle'], lib="torch", suffix=0)
