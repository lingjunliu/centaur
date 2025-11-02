
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def take_inputs():
    list_of_inputs = []

    # Input 1
    input_arr = torch.tensor([[4, 3, 5], [6, 7, 8]]).numpy()
    index_arr = torch.tensor([0, 2, 5], dtype=torch.long).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "index": index_arr}))

    # Input 2
    input_arr = torch.arange(12, dtype=torch.int32).reshape(2, 2, 3).numpy()
    index_arr = torch.tensor([[0, 7], [5, 10]], dtype=torch.long).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "index": index_arr}))

    # Input 3
    input_arr = torch.tensor([-1.0, -2.5, 3.3, 0.0, 4.4], dtype=torch.float32).numpy()
    index_arr = torch.tensor([0, 3, 4], dtype=torch.long).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "index": index_arr}))

    # Input 4
    input_arr = torch.tensor([[True, False], [False, True]], dtype=torch.bool).numpy()
    index_arr = torch.tensor([0, 3, 1, 2], dtype=torch.long).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "index": index_arr}))

    # Input 5
    input_arr = torch.tensor(
        [[1+2j, 3-4j, 5+0j],
         [0+1j, -2-2j, 7+7j],
         [9+0j, -1+1j, 2-3j]], dtype=torch.complex64
    ).numpy()
    index_arr = torch.tensor([8, 0, 4, 2, 6], dtype=torch.long).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "index": index_arr}))

    # Input 6
    input_arr = torch.tensor([1.5, -2.5, 3.0, 4.5], dtype=torch.float16).numpy()
    index_arr = torch.tensor(1, dtype=torch.long).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "index": index_arr}))

    # Input 7
    input_arr = torch.arange(120, dtype=torch.float32).reshape(2, 3, 4, 5).numpy()
    index_arr = torch.tensor([0, 59, 119, 24, 75], dtype=torch.long).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "index": index_arr}))

    # Input 8
    input_arr = torch.empty((0,), dtype=torch.float32).numpy()
    index_arr = torch.empty((0,), dtype=torch.long).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "index": index_arr}))

    # Input 9
    input_arr = torch.empty((1, 0, 2), dtype=torch.float32).numpy()
    index_arr = torch.empty((2, 0), dtype=torch.long).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "index": index_arr}))

    # Input 10
    input_arr = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.uint8).numpy()
    index_arr = torch.tensor([0, 1, 2, 3, 4, 7], dtype=torch.long).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "index": index_arr}))

    # Input 11
    input_arr = torch.linspace(0, 1, steps=5, dtype=torch.float64).numpy()
    index_arr = torch.tensor([[0, 2, 4], [1, 3, 0]], dtype=torch.long).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "index": index_arr}))

    # Input 12
    input_arr = torch.tensor(3+4j, dtype=torch.complex128).numpy()
    index_arr = torch.tensor([0], dtype=torch.long).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "index": index_arr}))

    return list_of_inputs

generated_inputs["torch.take"] = take_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.take' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.take'.")


check_valid('torch.take', generated_inputs['torch.take'], lib="torch", suffix=0)
