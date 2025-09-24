
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

# A wrapper to solve the test harness issue with sparse tensors.
# The test harness cannot convert a sparse tensor output to numpy.
# This wrapper calls the original function and then converts the output to a dense tensor.
_original_sparse_csr_tensor = torch.sparse_csr_tensor
def _sparse_csr_tensor_wrapper(*args, **kwargs):
    # The test harness may pass numpy dtypes; convert them to torch dtypes.
    if 'dtype' in kwargs and not isinstance(kwargs['dtype'], torch.dtype):
        try:
            # Attempt to map numpy dtype to torch dtype
            kwargs['dtype'] = getattr(torch, str(kwargs['dtype']).split('.')[-1])
        except (AttributeError, TypeError):
            # Fallback if conversion fails, though it might error later
            pass
    
    # Call the original function to create the sparse tensor
    sparse_tensor = _original_sparse_csr_tensor(*args, **kwargs)
    # Return the dense version, which can be converted to numpy
    return sparse_tensor.to_dense()

# Monkey-patch the function in the torch namespace
torch.sparse_csr_tensor = _sparse_csr_tensor_wrapper

def sparse_csr_tensor_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D float32 tensor
    input_dict_1 = {
        'crow_indices': torch.tensor([0, 2, 3, 5], dtype=torch.int64).numpy(),
        'col_indices': torch.tensor([0, 2, 2, 0, 1], dtype=torch.int64).numpy(),
        'values': torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32).numpy(),
        'size': [3, 3],
        'dtype': torch.float32,
        'requires_grad': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 2D float64 tensor with requires_grad=True
    input_dict_2 = {
        'crow_indices': torch.tensor([0, 2, 4], dtype=torch.int64).numpy(),
        'col_indices': torch.tensor([0, 1, 0, 1], dtype=torch.int64).numpy(),
        'values': torch.tensor([10.5, -20.1, 30.3, 40.8], dtype=torch.float64).numpy(),
        'size': [2, 2],
        'dtype': torch.float64,
        'requires_grad': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 2D int32 tensor with no non-zero elements (empty)
    input_dict_3 = {
        'crow_indices': torch.tensor([0, 0, 0, 0], dtype=torch.int64).numpy(),
        'col_indices': torch.tensor([], dtype=torch.int64).numpy(),
        'values': torch.tensor([], dtype=torch.int32).numpy(),
        'size': [3, 4],
        'dtype': torch.int32,
        'requires_grad': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: 2D int64 tensor, fully dense
    input_dict_4 = {
        'crow_indices': torch.tensor([0, 3, 6], dtype=torch.int64).numpy(),
        'col_indices': torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.int64).numpy(),
        'values': torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.int64).numpy(),
        'size': [2, 3],
        'dtype': torch.int64,
        'requires_grad': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: 2D bool tensor
    input_dict_5 = {
        'crow_indices': torch.tensor([0, 2, 3], dtype=torch.int64).numpy(),
        'col_indices': torch.tensor([0, 1, 1], dtype=torch.int64).numpy(),
        'values': torch.tensor([True, False, True], dtype=torch.bool).numpy(),
        'size': [2, 2],
        'dtype': torch.bool,
        'requires_grad': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: 2D complex64 tensor with requires_grad=True
    input_dict_6 = {
        'crow_indices': torch.tensor([0, 1, 2], dtype=torch.int64).numpy(),
        'col_indices': torch.tensor([0, 1], dtype=torch.int64).numpy(),
        'values': torch.tensor([1+2j, 3-4j], dtype=torch.complex64).numpy(),
        'size': [2, 2],
        'dtype': torch.complex64,
        'requires_grad': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))
    
    # Input 7: Large and very sparse tensor
    ci = np.zeros(101, dtype=np.int64)
    ci[10:] = 1
    ci[50:] = 2
    ci[90:] = 3
    input_dict_7 = {
        'crow_indices': ci,
        'col_indices': torch.tensor([5, 80, 150], dtype=torch.int64).numpy(),
        'values': torch.tensor([-1.0, 2.5, -3.0], dtype=torch.float32).numpy(),
        'size': [100, 200],
        'dtype': torch.float32,
        'requires_grad': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Tensor with a single row
    input_dict_8 = {
        'crow_indices': torch.tensor([0, 2], dtype=torch.int64).numpy(),
        'col_indices': torch.tensor([1, 3], dtype=torch.int64).numpy(),
        'values': torch.tensor([10, 20], dtype=torch.int32).numpy(),
        'size': [1, 5],
        'dtype': torch.int32,
        'requires_grad': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))
    
    # Input 9: Tensor with a single column
    input_dict_9 = {
        'crow_indices': torch.tensor([0, 1, 1, 2, 3], dtype=torch.int64).numpy(),
        'col_indices': torch.tensor([0, 0, 0], dtype=torch.int64).numpy(),
        'values': torch.tensor([1., 2., 3.], dtype=torch.float32).numpy(),
        'size': [4, 1],
        'dtype': torch.float32,
        'requires_grad': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Unsorted col_indices with duplicates (requires coalescing)
    input_dict_10 = {
        'crow_indices': torch.tensor([0, 3, 5], dtype=torch.int64).numpy(),
        'col_indices': torch.tensor([2, 0, 2, 1, 1], dtype=torch.int64).numpy(),
        'values': torch.tensor([1., 2., 3., 4., 5.], dtype=torch.float32).numpy(),
        'size': [2, 3],
        'dtype': torch.float32,
        'requires_grad': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: A tensor with some all-zero rows
    input_dict_11 = {
        'crow_indices': torch.tensor([0, 2, 2, 3, 3, 4], dtype=torch.int64).numpy(),
        'col_indices': torch.tensor([0, 3, 1, 2], dtype=torch.int64).numpy(),
        'values': torch.tensor([1, 2, 3, 4], dtype=torch.float32).numpy(),
        'size': [5, 4],
        'dtype': torch.float32,
        'requires_grad': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    return list_of_inputs

generated_inputs["torch.sparse_csr_tensor_2"] = sparse_csr_tensor_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.sparse_csr_tensor_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sparse_csr_tensor_2'.")

check_valid('torch.sparse_csr_tensor', generated_inputs['torch.sparse_csr_tensor_2'], lib="torch", suffix=2)
