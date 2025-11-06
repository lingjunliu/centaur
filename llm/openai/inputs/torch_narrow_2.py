
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def narrow_inputs():
    list_of_inputs = []

    input_arr = torch.tensor([[1., 2., 3.],
                              [4., 5., 6.],
                              [7., 8., 9.]], dtype=torch.float32).numpy()
    dim = 0
    start = torch.tensor(0, dtype=torch.int64).numpy()
    length = 2
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "start": start, "length": length}))

    input_arr = torch.arange(9, dtype=torch.int64).reshape(3, 3).numpy()
    dim = 1
    start = torch.tensor(1, dtype=torch.int64).numpy()
    length = 2
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "start": start, "length": length}))

    input_arr = torch.tensor([[1., 2., 3.],
                              [4., 5., 6.],
                              [7., 8., 9.]], dtype=torch.float64).numpy()
    dim = -1
    start = torch.tensor(-1, dtype=torch.int64).numpy()
    length = 1
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "start": start, "length": length}))

    input_arr = torch.linspace(0, 1, 5, dtype=torch.float32).numpy()
    dim = 0
    start = torch.tensor(2, dtype=torch.int64).numpy()
    length = 3
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "start": start, "length": length}))

    input_arr = torch.arange(5, dtype=torch.float64).numpy()
    dim = 0
    start = torch.tensor(-3, dtype=torch.int64).numpy()
    length = 2
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "start": start, "length": length}))

    input_arr = torch.randn(2, 3, 4, dtype=torch.float32).numpy()
    dim = 2
    start = torch.tensor(1, dtype=torch.int64).numpy()
    length = 2
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "start": start, "length": length}))

    input_arr = torch.arange(24, dtype=torch.int32).reshape(2, 3, 4).numpy()
    dim = -3
    start = torch.tensor(1, dtype=torch.int64).numpy()
    length = 1
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "start": start, "length": length}))

    input_arr = torch.randn(2, 3, 4, 5, dtype=torch.float32).numpy()
    dim = 3
    start = torch.tensor(0, dtype=torch.int64).numpy()
    length = 0
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "start": start, "length": length}))

    input_arr = torch.tensor([[[[True], [False], [True]]],
                               [[[False], [True], [False]]]], dtype=torch.bool).numpy()
    dim = -1
    start = torch.tensor(0, dtype=torch.int64).numpy()
    length = 1
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "start": start, "length": length}))

    input_arr = torch.arange(1*2*3*4*5, dtype=torch.float16).reshape(1, 2, 3, 4, 5).numpy()
    dim = 4
    start = torch.tensor(-5, dtype=torch.int64).numpy()
    length = 5
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "start": start, "length": length}))

    input_arr = torch.randn(2, 2, 2, dtype=torch.complex64).numpy()
    dim = 1
    start = torch.tensor(0, dtype=torch.int64).numpy()
    length = 1
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "start": start, "length": length}))

    input_arr = torch.arange(20, dtype=torch.float64).reshape(4, 5).numpy()
    dim = -2
    start = torch.tensor(-2, dtype=torch.int64).numpy()
    length = 2
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "start": start, "length": length}))

    return list_of_inputs

generated_inputs["torch.narrow_2"] = narrow_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.narrow_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.narrow_2'.")


check_valid('torch.narrow', generated_inputs['torch.narrow_2'], lib="torch", suffix=2)
