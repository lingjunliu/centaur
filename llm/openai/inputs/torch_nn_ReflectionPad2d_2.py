
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def reflectionpad2d_inputs():
    list_of_inputs = []

    # 1
    input_arr = torch.arange(9, dtype=torch.float32).reshape(1, 1, 3, 3).numpy()
    padding = (2, 2, 2, 2)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    # 2
    input_arr = torch.randn(2, 3, 5, 4, dtype=torch.float64).numpy()
    padding = (1, 0, 2, 1)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    # 3
    input_arr = torch.arange(3*4*6, dtype=torch.float32).reshape(3, 4, 6).numpy()
    padding = (0, 0, 1, 1)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    # 4
    input_arr = torch.tensor([[[1, 2],
                               [3, 4]]], dtype=torch.int64).numpy()  # (C=1,H=2,W=2)
    padding = (1, 1, 1, 1)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    # 5
    input_arr = torch.randint(0, 256, (2, 1, 7, 5), dtype=torch.uint8).numpy()
    padding = (3, 2, 0, 4)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    # 6
    input_arr = torch.randn(2, 8, 3, dtype=torch.float16).numpy()
    padding = (2, 0, 0, 7)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    # 7
    real = torch.randn(1, 2, 4, 4, dtype=torch.float32)
    imag = torch.randn(1, 2, 4, 4, dtype=torch.float32)
    input_arr = (real + 1j * imag).numpy()
    padding = (1, 3, 2, 0)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    # 8
    input_arr = torch.randint(-100, 100, (4, 3, 10, 10), dtype=torch.int32).numpy()
    padding = (0, 0, 0, 0)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    # 9
    input_arr = torch.linspace(0, 1, steps=4*3*10, dtype=torch.float32).reshape(4, 3, 10).numpy()
    padding = (9, 0, 2, 0)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    # 10
    input_arr = torch.randn(1, 5, 32, 1, dtype=torch.float32).numpy()
    padding = (0, 0, 15, 0)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    # 11
    input_arr = torch.arange(9, dtype=torch.float64).reshape(1, 1, 9).numpy()  # (C=1,H=1,W=9) incorrect shape; fix to (C,H,W)
    input_arr = torch.arange(9, dtype=torch.float64).reshape(1, 1, 9).expand(1, 1, 9).numpy()  # still (1,1,9)
    # Ensure (C,H,W)
    input_arr = torch.arange(9, dtype=torch.float64).reshape(1, 9).unsqueeze(0).numpy()  # (1,1,9)
    # Construct directly as (C=1,H=1,W=9)
    input_arr = np.arange(9, dtype=np.float64).reshape(1, 1, 9)
    padding = (2, 2, 0, 0)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    # 12
    input_arr = torch.randint(-5, 6, (3, 2, 2, 5), dtype=torch.int64).numpy()
    padding = (4, 0, 1, 0)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    return list_of_inputs

generated_inputs["torch.nn.ReflectionPad2d_2"] = reflectionpad2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.ReflectionPad2d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ReflectionPad2d_2'.")


check_valid('torch.nn.ReflectionPad2d', generated_inputs['torch.nn.ReflectionPad2d_2'], lib="torch", suffix=2)
