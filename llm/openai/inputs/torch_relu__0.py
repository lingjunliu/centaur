
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def relu_inplace_inputs():
    list_of_inputs = []

    input = torch.tensor([-3.0, 0.0, 2.5, -1.2, 4.3], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[-1.0, 2.0, -3.0],
                          [4.5, -5.5, 0.0]], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[[-3, -2, -1],
                           [0, 1, 2]],
                          [[3, -4, 5],
                           [-6, 7, -8]]], dtype=torch.int64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[-1.5, 0.0, 2.0],
                          [3.3, -4.1, 5.5]], dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor(-3.5, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([float('nan'), float('inf'), -float('inf'), -0.0, 0.0, 1.0], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    base = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    input = base.permute(1, 0, 2).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    mat = torch.tensor([[-1, 2, -3],
                        [4, -5, 6]], dtype=torch.int32)
    input = mat.t().numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[0, 1, 255],
                          [128, 64, 0]], dtype=torch.uint8).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[-128, -1, 0],
                          [1, 127, -5]], dtype=torch.int8).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.zeros((2, 0, 3), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.arange(-24, 24, dtype=torch.int16).reshape(2, 2, 3, 4).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    return list_of_inputs

generated_inputs["torch.relu_"] = relu_inplace_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.relu_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.relu_'.")


check_valid('torch.relu_', generated_inputs['torch.relu_'], lib="torch", suffix=0)
