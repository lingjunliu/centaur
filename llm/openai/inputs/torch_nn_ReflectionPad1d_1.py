
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def reflectionpad1d_inputs():
    list_of_inputs = []

    # 1: 2D float32, padding 0
    input_arr = torch.arange(15, dtype=torch.float32).reshape(3, 5).numpy()
    input_dict = {"padding": 0, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 2: 3D float32, padding 1
    input_arr = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4).numpy()
    input_dict = {"padding": 1, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 3: 3D float64, padding 2
    input_arr = torch.randn(2, 3, 7, dtype=torch.float64).numpy()
    input_dict = {"padding": 2, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 4: 2D int32, padding 3
    input_arr = torch.arange(40, dtype=torch.int32).reshape(4, 10).numpy()
    input_dict = {"padding": 3, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 5: 3D int64, padding 4
    input_arr = torch.arange(18, dtype=torch.int64).reshape(2, 1, 9).numpy()
    input_dict = {"padding": 4, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 6: 2D float16, padding 5
    input_arr = torch.linspace(-1, 1, steps=11, dtype=torch.float16).unsqueeze(0).numpy()
    input_dict = {"padding": 5, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 7: 3D complex64, padding 6
    real = torch.randn(3, 2, 13, dtype=torch.float32)
    imag = torch.randn(3, 2, 13, dtype=torch.float32)
    input_arr = (real + 1j * imag).numpy()
    input_dict = {"padding": 6, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 8: 3D float32 with NaN/Inf, padding 7
    input_arr = torch.randn(1, 3, 20, dtype=torch.float32).numpy()
    input_arr[0, 0, 0] = np.nan
    input_arr[0, 1, 1] = np.inf
    input_arr[0, 2, 2] = -np.inf
    input_dict = {"padding": 7, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 9: 2D float64 small width, padding 1
    input_arr = torch.tensor([[-1.0, 2.0], [3.5, -4.5]], dtype=torch.float64).numpy()
    input_dict = {"padding": 1, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 10: 3D float32 larger, padding 8
    input_arr = (torch.ones(4, 5, 17, dtype=torch.float32) * 3.14).numpy()
    input_dict = {"padding": 8, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 11: 2D float32 minimal width 3, padding 2
    input_arr = torch.tensor([[0.5, -1.5, 2.5]], dtype=torch.float32).numpy()
    input_dict = {"padding": 2, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 12: 3D float32, padding 9
    input_arr = torch.arange(40, dtype=torch.float32).reshape(2, 2, 10).numpy()
    input_dict = {"padding": 9, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.ReflectionPad1d_1"] = reflectionpad1d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.ReflectionPad1d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ReflectionPad1d_1'.")


check_valid('torch.nn.ReflectionPad1d', generated_inputs['torch.nn.ReflectionPad1d_1'], lib="torch", suffix=1)
