
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy

def adaptive_max_pool1d_inputs():
    list_of_inputs = []

    input1 = torch.randn(1, 3, 20).numpy()
    output_size1 = 5
    input_dict1 = {"input": input1, "output_size": output_size1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(2, 4, 15).numpy()
    output_size2 = 3
    input_dict2 = {"input": input2, "output_size": output_size2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randint(0, 10, (1, 2, 30)).float().numpy()
    output_size3 = 10
    input_dict3 = {"input": input3, "output_size": output_size3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(3, 1, 25).numpy()
    output_size4 = 7
    input_dict4 = {"input": input4, "output_size": output_size4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(1, 5, 12).numpy()
    output_size5 = 2
    input_dict5 = {"input": input5, "output_size": output_size5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = torch.randn(1, 1, 1).numpy()
    output_size6 = 1
    input_dict6 = {"input": input6, "output_size": output_size6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    return list_of_inputs

generated_inputs["torch.nn.functional.adaptive_max_pool1d_1"] = adaptive_max_pool1d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.adaptive_max_pool1d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.adaptive_max_pool1d_1'.")

check_valid('torch.nn.functional.adaptive_max_pool1d', generated_inputs['torch.nn.functional.adaptive_max_pool1d_1'], lib="torch")
