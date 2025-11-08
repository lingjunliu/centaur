
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def relu_inputs():
    list_of_inputs = []

    # 1
    input = torch.tensor([-3.0, 0.0, 2.5], dtype=torch.float32).numpy()
    input_dict = {"input": input, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 2
    input = torch.tensor([[-1.0, 2.0, -3.5],
                          [4.2, -0.1, 0.0]], dtype=torch.float64).numpy()
    input_dict = {"input": input, "inplace": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 3
    input = (torch.randn(2, 2, 3, dtype=torch.float16) * 2 - 1).numpy()
    input_dict = {"input": input, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 4
    input = torch.randn(1, 3, 4, 4, dtype=torch.float32).numpy()
    input_dict = {"input": input, "inplace": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 5
    input = torch.tensor(-5.0, dtype=torch.float32).numpy()
    input_dict = {"input": input, "inplace": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 6
    input = torch.empty((0,), dtype=torch.float32).numpy()
    input_dict = {"input": input, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 7
    input = torch.tensor([float('inf'), float('-inf'), float('nan'),
                          -0.0, 0.0, 1e-30, -1e-30], dtype=torch.float32).numpy()
    input_dict = {"input": input, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 8
    base = torch.arange(24.0, dtype=torch.float32).reshape(4, 6)
    input = (base.t() - 12.0).numpy()
    input_dict = {"input": input, "inplace": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 9
    input = np.linspace(-3.0, 3.0, num=7, dtype=np.float32)
    input_dict = {"input": input, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 10
    input = torch.tensor([[[[[-1.0, 0.5, 2.0]]],
                            [[[3.0, -2.0, 0.0]]]]], dtype=torch.float32).repeat(1,1,1,1,1).numpy()
    input_dict = {"input": input, "inplace": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 11
    input = torch.tensor([-65504.0, 65504.0, -1e-4, 1e-4], dtype=torch.float16).numpy()
    input_dict = {"input": input, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 12
    input = torch.zeros((2, 0, 4), dtype=torch.float32).numpy()
    input_dict = {"input": input, "inplace": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.functional.relu"] = relu_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.relu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.relu'.")


check_valid('torch.nn.functional.relu', generated_inputs['torch.nn.functional.relu'], lib="torch", suffix=0)
