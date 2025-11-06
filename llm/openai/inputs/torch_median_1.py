
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def median_inputs():
    list_of_inputs = []
    
    # 1) 0-D float64
    input_arr = torch.tensor(3.14, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr}))
    
    # 2) 1-D float32 even length
    input_arr = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr}))
    
    # 3) 1-D int64 odd length with negatives
    input_arr = torch.tensor([5, -1, 0, 8, 2], dtype=torch.int64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr}))
    
    # 4) 2-D float32 random
    input_arr = torch.randn(2, 3, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr}))
    
    # 5) 2-D int16 with negatives and positives
    input_arr = torch.tensor([[-2, -3, 4], [7, 8, -9]], dtype=torch.int16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr}))
    
    # 6) 3-D float16 linspace
    input_arr = torch.linspace(-1, 1, steps=24, dtype=torch.float16).reshape(2, 3, 4).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr}))
    
    # 7) 1-D float32 non-contiguous slice
    input_arr = torch.arange(20, dtype=torch.float32)[::2].numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr}))
    
    # 8) 4-D float64 arange
    input_arr = torch.arange(2*3*4*5, dtype=torch.float64).reshape(2, 3, 4, 5).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr}))
    
    # 9) 1-D float32 with NaN
    input_arr = torch.tensor([1.0, float('nan'), 2.0, 3.0], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr}))
    
    # 10) 1-D float32 with infinities
    input_arr = torch.tensor([float('-inf'), -1.0, 0.0, float('inf')], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr}))
    
    # 11) 1-D int8 extremes
    input_arr = torch.tensor([-128, -128, 127, 0, 1], dtype=torch.int8).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr}))
    
    # 12) 3-D float32 with repeated values
    input_arr = torch.full((2, 2, 2), 7.0, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr}))
    
    return list_of_inputs

generated_inputs["torch.median_1"] = median_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.median_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.median_1'.")


check_valid('torch.median', generated_inputs['torch.median_1'], lib="torch", suffix=1)
