
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def smm_inputs():
    list_of_inputs = []

    # Input 1: Basic valid case
    input = torch.sparse_coo_tensor(indices=torch.tensor([[0, 1], [1, 0]]), values=torch.tensor([1.0, 2.0]), size=(2, 2))
    mat = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    input_dict = {"input": input.to_dense().numpy(), "mat": mat.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Larger matrices
    input = torch.sparse_coo_tensor(indices=torch.tensor([[0, 1, 2], [1, 2, 0]]), values=torch.tensor([1.0, 2.0, 3.0]), size=(3, 3))
    mat = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    input_dict = {"input": input.to_dense().numpy(), "mat": mat.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Sparse matrix with negative values
    input = torch.sparse_coo_tensor(indices=torch.tensor([[0, 1], [1, 0]]), values=torch.tensor([-1.0, 2.0]), size=(2, 2))
    mat = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    input_dict = {"input": input.to_dense().numpy(), "mat": mat.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Non-square matrices
    input = torch.sparse_coo_tensor(indices=torch.tensor([[0, 1], [1, 0]]), values=torch.tensor([1.0, 2.0]), size=(2, 3))
    mat = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    input_dict = {"input": input.to_dense().numpy(), "mat": mat.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different values and sizes.
    input = torch.sparse_coo_tensor(indices=torch.tensor([[0, 2], [1, 0]]), values=torch.tensor([4.0, -2.0]), size=(3, 2))
    mat = torch.tensor([[-1.0, 0.5], [3.0, -2.0], [0.0,1.0]])
    input_dict = {"input": input.to_dense().numpy(), "mat": mat.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.smm"] = smm_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.smm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.smm'.")

check_valid('torch.smm', generated_inputs['torch.smm'], lib="torch", suffix=0)
