
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def index_select_inputs():
    list_of_inputs = []

    # Input 1: 2D float tensor, positive index
    input_tensor = torch.randn(3, 4).numpy()
    index_tensor = torch.tensor([0, 2]).numpy()
    dim = 0
    input_dict = {"input": input_tensor, "dim": dim, "index": index_tensor, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3D int tensor, positive index
    input_tensor = torch.randint(0, 10, (2, 3, 5)).numpy()
    index_tensor = torch.tensor([0, 1]).numpy() # changed negative indices to positive
    dim = 1
    input_dict = {"input": input_tensor, "dim": dim, "index": index_tensor, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D tensor, index within bounds
    input_tensor = torch.arange(5).float().numpy()
    index_tensor = torch.tensor([2]).numpy()
    dim = 0
    input_dict = {"input": input_tensor, "dim": dim, "index": index_tensor, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 4D complex tensor, LongTensor index
    input_tensor = torch.randn(2, 2, 2, 2, dtype=torch.complex64).numpy()
    index_tensor = torch.tensor([0, 1]).long().numpy()
    dim = 2
    input_dict = {"input": input_tensor, "dim": dim, "index": index_tensor, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D tensor, dim = -1
    input_tensor = torch.randn(5, 5).numpy()
    index_tensor = torch.tensor([1, 3]).numpy()
    dim = -1
    input_dict = {"input": input_tensor, "dim": dim, "index": index_tensor, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.index_select"] = index_select_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.index_select' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.index_select'.")

check_valid('torch.index_select', generated_inputs['torch.index_select'], lib="torch")
