
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def argmax_inputs():
    list_of_inputs = []

    input = torch.tensor([1.0, -2.5, 3.3, 0.0], dtype=torch.float32).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.tensor([[1.3398, 0.2663, -0.2686, 0.2450],
                          [-0.7401, -0.8805, -0.3402, -1.1936],
                          [0.4907, -1.3948, -1.0691, -0.3132],
                          [-1.6092, 0.5419, -0.2993, 0.3195]], dtype=torch.float64).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.arange(24, dtype=torch.int64).view(2, 3, 4).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(2, 2, 3, 4, dtype=torch.float32).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.tensor(-5, dtype=torch.int32).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.tensor([[float('-inf'), -1.0, 0.0],
                          [1.0, float('inf'), 2.0]], dtype=torch.float32).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = torch.arange(20, dtype=torch.int64).view(4, 5)[:, ::2]
    input = x.numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.tensor([[-30000, -1, -2],
                          [-123, -32768, -5]], dtype=torch.int16).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.tensor([[[5, 5], [1, 2]],
                          [[5, 0], [5, 3]],
                          [[4, 4], [4, 4]]], dtype=torch.int64).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.tensor([0, 255, 10, 255, 100], dtype=torch.uint8).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(2, 1, 3, 1, 4, dtype=torch.float32).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.linspace(-10, 10, steps=21, dtype=torch.float32).reshape(3, 7).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.argmax_1"] = argmax_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.argmax_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.argmax_1'.")


check_valid('torch.argmax', generated_inputs['torch.argmax_1'], lib="torch", suffix=1)
