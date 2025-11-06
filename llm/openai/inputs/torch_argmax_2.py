
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def argmax_inputs():
    list_of_inputs = []

    input_arr = torch.tensor([1.0, -2.0, 3.5, 3.5], dtype=torch.float32).numpy()
    dim = 0
    keepdim = False
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim}))

    input_arr = torch.tensor([[1.0, 2.0, -3.0],
                              [4.0, 5.0, 6.0]], dtype=torch.float64).numpy()
    dim = 1
    keepdim = True
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim}))

    input_arr = torch.tensor([[10, 2, 3],
                              [0, -1, -2],
                              [4, 5, 6]], dtype=torch.int64).numpy()
    dim = 0
    keepdim = False
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim}))

    input_arr = torch.randn(2, 3, 4, dtype=torch.float32).numpy()
    dim = -1
    keepdim = True
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim}))

    input_arr = torch.randint(0, 256, (3, 4, 2), dtype=torch.uint8).numpy()
    dim = 1
    keepdim = False
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim}))

    input_arr = torch.randn(2, 2, 3, 3, dtype=torch.float32).numpy()
    dim = 2
    keepdim = True
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim}))

    input_arr = torch.randint(-100, 100, (2, 3, 4, 5), dtype=torch.int32).numpy()
    dim = -2
    keepdim = True
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim}))

    input_arr = torch.tensor([[0.0, 0.0, 1.0],
                              [2.0, 2.0, 2.0],
                              [-1.0, -1.0, -1.0]], dtype=torch.float32).numpy()
    dim = -1
    keepdim = False
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim}))

    input_arr = torch.randn(1, 2, 1, 3, 4, dtype=torch.float32).numpy()
    dim = 3
    keepdim = True
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim}))

    input_arr = torch.tensor([[-128, -3, -3, 127, 0],
                              [10, 10, 10, -5, -5],
                              [7, 8, 9, 10, 11],
                              [0, 0, 0, 0, 0]], dtype=torch.int8).numpy()
    dim = 0
    keepdim = False
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim}))

    input_arr = torch.linspace(-1, 1, steps=10, dtype=torch.float32).numpy()
    dim = -1
    keepdim = True
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim}))

    input_arr = torch.randint(-32768, 32767, (2, 3, 3), dtype=torch.int16).numpy()
    dim = 2
    keepdim = False
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim}))

    return list_of_inputs

generated_inputs["torch.argmax_2"] = argmax_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.argmax_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.argmax_2'.")


check_valid('torch.argmax', generated_inputs['torch.argmax_2'], lib="torch", suffix=2)
