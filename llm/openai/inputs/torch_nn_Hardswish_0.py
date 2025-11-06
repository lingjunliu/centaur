
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def hardswish_inputs():
    list_of_inputs = []
    
    # 1: scalar float32
    input_arr = torch.tensor(4.5, dtype=torch.float32).numpy()
    input_dict = {"inplace": False, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 2: 1D array with thresholds
    input_arr = torch.tensor([-5.0, -3.0, -1.0, 0.0, 2.0, 3.0, 5.0], dtype=torch.float32).numpy()
    input_dict = {"inplace": True, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 3: 2D float64
    input_arr = torch.linspace(-5, 5, steps=12, dtype=torch.float64).reshape(3, 4).numpy()
    input_dict = {"inplace": False, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 4: 3D float16 uniform
    input_arr = torch.empty(2, 3, 4, dtype=torch.float16).uniform_(-6, 6).numpy()
    input_dict = {"inplace": True, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 5: 4D float32 normal
    input_arr = torch.randn(2, 1, 3, 3, dtype=torch.float32).numpy()
    input_dict = {"inplace": False, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 6: empty (0,) float32
    input_arr = torch.tensor([], dtype=torch.float32).numpy()
    input_dict = {"inplace": True, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 7: empty interior (2,0,3) float32
    input_arr = torch.empty(2, 0, 3, dtype=torch.float32).numpy()
    input_dict = {"inplace": False, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 8: very large/small values
    input_arr = torch.tensor([1e-6, 1e6, -1e6, 6.0, -6.0], dtype=torch.float32).numpy()
    input_dict = {"inplace": True, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 9: NaN and Inf values float64
    input_arr = torch.tensor([float('nan'), float('inf'), float('-inf'), -2.5, 2.5], dtype=torch.float64).numpy()
    input_dict = {"inplace": False, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 10: non-contiguous transpose
    input_arr = torch.arange(12, dtype=torch.float32).reshape(3, 4).t().numpy()
    input_dict = {"inplace": True, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 11: typical conv batch shape
    input_arr = torch.randn(1, 3, 4, 4, dtype=torch.float32).numpy()
    input_dict = {"inplace": False, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 12: negative values below -3
    input_arr = torch.full((5,), -4.0, dtype=torch.float32).numpy()
    input_dict = {"inplace": True, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 13: exact thresholds in 2D
    input_arr = torch.tensor([[-3.0, 3.0], [-3.0, 3.0]], dtype=torch.float32).numpy()
    input_dict = {"inplace": False, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 14: 5D float16
    input_arr = torch.randn(1, 2, 1, 2, 3, dtype=torch.float16).numpy()
    input_dict = {"inplace": True, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 15: 1D float64 random
    input_arr = torch.randn(7, dtype=torch.float64).numpy()
    input_dict = {"inplace": False, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.Hardswish"] = hardswish_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.Hardswish' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Hardswish'.")


check_valid('torch.nn.Hardswish', generated_inputs['torch.nn.Hardswish'], lib="torch", suffix=0)
