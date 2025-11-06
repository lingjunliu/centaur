
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def squeeze_inputs():
    list_of_inputs = []

    # 1
    input_arr = torch.tensor([1.0, -2.5, 3.3], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr}))

    # 2
    input_arr = torch.ones((1, 4), dtype=torch.int64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr}))

    # 3
    input_arr = torch.tensor([[[1.0]], [[-7.0]]], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr}))

    # 4
    input_arr = ((torch.arange(15) % 2) == 0).reshape(1, 3, 1, 5).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr}))

    # 5
    input_arr = torch.randn(2, 1, 3, 1, 1, dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr}))

    # 6
    input_arr = torch.tensor(42.0).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr}))

    # 7
    input_arr = torch.zeros((1, 1, 1, 1, 1, 1), dtype=torch.int8).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr}))

    # 8
    input_arr = torch.tensor([[[-1, 0, 1], [2, -3, 4]]], dtype=torch.int32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr}))

    # 9
    real = torch.randn(2, 1, 2, dtype=torch.float32)
    imag = torch.randn(2, 1, 2, dtype=torch.float32)
    input_arr = (real + 1j * imag).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr}))

    # 10
    input_arr = torch.arange(4 * 1 * 5 * 1 * 6, dtype=torch.float32).reshape(4, 1, 5, 1, 6).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr}))

    # 11
    input_arr = torch.tensor([255], dtype=torch.uint8).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr}))

    # 12
    input_arr = torch.randn(2, 0, 1, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr}))

    return list_of_inputs

generated_inputs["torch.squeeze_1"] = squeeze_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.squeeze_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.squeeze_1'.")


check_valid('torch.squeeze', generated_inputs['torch.squeeze_1'], lib="torch", suffix=1)
