
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy

def householder_product_inputs():
    list_of_inputs = []

    # Helper function to create an input dictionary
    # The inputs to householder_product (input, h) are the outputs of torch.geqrf (A, tau)
    # The constraint is that for input tensor shape (..., m, n), m must be >= n.
    def create_input_dict(tensor):
        input_tensor, h_tensor = torch.geqrf(tensor)
        return {
            'input': input_tensor.numpy(),
            'h': h_tensor.numpy()
        }

    # Case 1: Basic case, m > n, float32
    list_of_inputs.append(copy.deepcopy(create_input_dict(torch.randn(5, 3, dtype=torch.float32))))

    # Case 2: Square matrix, m = n, float32
    list_of_inputs.append(copy.deepcopy(create_input_dict(torch.randn(4, 4, dtype=torch.float32))))

    # Case 3: Batched input, single batch dim, m > n
    list_of_inputs.append(copy.deepcopy(create_input_dict(torch.randn(2, 5, 4, dtype=torch.float32))))

    # Case 4: Batched input, multiple batch dims, m > n
    list_of_inputs.append(copy.deepcopy(create_input_dict(torch.randn(2, 3, 6, 4, dtype=torch.float32))))

    # Case 5: float64 dtype, tall matrix
    list_of_inputs.append(copy.deepcopy(create_input_dict(torch.randn(7, 4, dtype=torch.float64))))

    # Case 6: complex64 dtype, m > n
    list_of_inputs.append(copy.deepcopy(create_input_dict(torch.randn(5, 2, dtype=torch.complex64))))

    # Case 7: complex128 dtype, batched square matrix
    list_of_inputs.append(copy.deepcopy(create_input_dict(torch.randn(3, 4, 4, dtype=torch.complex128))))

    # Case 8: Input with negative values, m > n
    list_of_inputs.append(copy.deepcopy(create_input_dict(-torch.rand(6, 4, dtype=torch.float32))))

    # Case 9: Batched input with m = n, complex64
    list_of_inputs.append(copy.deepcopy(create_input_dict(torch.randn(3, 3, 3, dtype=torch.complex64))))

    # Case 10: Another tall and skinny matrix, float64
    list_of_inputs.append(copy.deepcopy(create_input_dict(torch.randn(10, 2, dtype=torch.float64))))
    
    # Case 11: Minimal square matrix 2x2
    list_of_inputs.append(copy.deepcopy(create_input_dict(torch.randn(2, 2, dtype=torch.float32))))
    
    # Case 12: Edge case where n=1
    list_of_inputs.append(copy.deepcopy(create_input_dict(torch.randn(5, 1, dtype=torch.float32))))

    return list_of_inputs

generated_inputs["torch.linalg.householder_product"] = householder_product_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.linalg.householder_product' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.householder_product'.")

check_valid('torch.linalg.householder_product', generated_inputs['torch.linalg.householder_product'], lib="torch", suffix=0)
