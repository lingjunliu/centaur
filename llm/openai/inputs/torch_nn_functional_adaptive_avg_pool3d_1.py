
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def adaptive_avg_pool3d_inputs():
    list_of_inputs = []

    input = torch.randn(2, 3, 4, 5, 6, dtype=torch.float32).numpy()
    output_size = (1, 1, 1)
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    input = (torch.randn(1, 1, 3, 3, 3, dtype=torch.float64) * -3.0).numpy()
    output_size = (2, 2, 2)
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    input = torch.randn(2, 5, 4, 7, dtype=torch.float32).numpy()
    output_size = (3, 2, 5)
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    t = torch.linspace(-10, 10, steps=2 * 1 * 2 * 3 * 4, dtype=torch.float64).reshape(2, 1, 2, 3, 4)
    input = t.numpy()
    output_size = (1, 3, 2)
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    input = torch.randn(4, 2, 2, 2, 2, dtype=torch.float16).numpy()
    output_size = (2, 1, 1)
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    base = torch.randn(1, 3, 4, 4, 4, dtype=torch.float32).numpy()
    input = base[..., ::-1]
    output_size = (1, 2, 3)
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    input = torch.tensor([[[[[5.0]]]]], dtype=torch.float32).numpy()
    output_size = (1, 1, 1)
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    input = torch.randn(1, 2, 2, 2, 2, dtype=torch.float32).numpy()
    output_size = (3, 4, 5)
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    input = torch.randn(1, 8, 9, 10, dtype=torch.float32).numpy()
    output_size = (4, 4, 4)
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    input = torch.randn(3, 5, 6, 7, 8, dtype=torch.float64).numpy()
    output_size = (6, 7, 8)
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    base = torch.randn(2, 3, 6, 6, 6, dtype=torch.float32).numpy()
    input = base[:, :, ::2, 1::2, ::-2]
    output_size = (2, 2, 2)
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    input = (torch.randn(5, 7, 3, 5, 4, dtype=torch.float32) - 1.5).numpy()
    output_size = (2, 3, 1)
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    return list_of_inputs

generated_inputs["torch.nn.functional.adaptive_avg_pool3d_1"] = adaptive_avg_pool3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.adaptive_avg_pool3d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.adaptive_avg_pool3d_1'.")


check_valid('torch.nn.functional.adaptive_avg_pool3d', generated_inputs['torch.nn.functional.adaptive_avg_pool3d_1'], lib="torch", suffix=1)
