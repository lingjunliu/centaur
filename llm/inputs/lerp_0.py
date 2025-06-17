
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def lerp_inputs():
    list_of_inputs = []

    # Example 1: Basic float tensors with scalar weight
    start = torch.arange(1., 5.).numpy()
    end = torch.full((4,), 10.).numpy()
    weight = 0.5
    input_dict = {"input": start, "end": end, "weight": weight, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2: Float tensors with tensor weight
    start = torch.arange(1., 5.).numpy()
    end = torch.full((4,), 10.).numpy()
    weight = torch.full_like(torch.from_numpy(start), 0.5).numpy()
    input_dict = {"input": start, "end": end, "weight": weight, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 3: Int tensors with scalar weight - Converting to float
    start = torch.arange(1, 5).float().numpy()
    end = torch.full((4,), 10.).numpy()
    weight = 0.25
    input_dict = {"input": start, "end": end, "weight": weight, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 4: Broadcasting with scalar weight
    start = torch.arange(1., 5.).reshape(2, 2).numpy()
    end = torch.full((2, 2), 10.).numpy()
    weight = 0.75
    input_dict = {"input": start, "end": end, "weight": weight, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 5: Broadcasting with tensor weight
    start = torch.arange(1., 5.).reshape(2, 2).numpy()
    end = torch.full((2, 2), 10.).numpy()
    weight = torch.full_like(torch.from_numpy(start), 0.3).numpy()
    input_dict = {"input": start, "end": end, "weight": weight, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Example 6: Negative values and weight > 1
    start = torch.arange(-2., 2.).numpy()
    end = torch.full((4,), 5.).numpy()
    weight = 1.5
    input_dict = {"input": start, "end": end, "weight": weight, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 7: Weight < 0
    start = torch.arange(1., 5.).numpy()
    end = torch.full((4,), 10.).numpy()
    weight = -0.5
    input_dict = {"input": start, "end": end, "weight": weight, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Example 8: 3D tensors
    start = torch.randn(2, 3, 4).numpy()
    end = torch.randn(2, 3, 4).numpy()
    weight = 0.6
    input_dict = {"input": start, "end": end, "weight": weight, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.lerp"] = lerp_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.lerp' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.lerp'.")

check_valid('torch.lerp', generated_inputs['torch.lerp'], lib="torch")
