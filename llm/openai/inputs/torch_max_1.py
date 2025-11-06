
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def torch_max_inputs():
    list_of_inputs = []

    input = torch.tensor([1.5, -2.3, 0.0, 4.2], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[0.0, -1.0, 2.5],
                          [3.3, -4.4, 5.5]], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.arange(-12, 12, dtype=torch.int64).reshape(2, 3, 4).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([True, False, True, True, False], dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor(3.14159265, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[1, -2, 3],
                          [-4, 5, -6],
                          [7, -8, 9]], dtype=torch.int32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[1.0, float('nan')],
                          [float('inf'), -float('inf')]], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[[-1.0, -0.5],
                           [0.1, 0.2]],
                          [[10.0, -10.0],
                           [5.0, 6.0]]], dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.arange(0, 16, dtype=torch.uint8).reshape(4, 4).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    t = torch.tensor([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0]], dtype=torch.float64)
    input = t.t().contiguous().numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.linspace(-1000, 1000, steps=101, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[[[1, 2], [3, 4]],
                           [[-5, -6], [7, 8]]],
                          [[[9, -10], [11, -12]],
                           [[13, 14], [-15, 16]]]], dtype=torch.int16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    return list_of_inputs

generated_inputs["torch.max_1"] = torch_max_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.max_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.max_1'.")


check_valid('torch.max', generated_inputs['torch.max_1'], lib="torch", suffix=1)
