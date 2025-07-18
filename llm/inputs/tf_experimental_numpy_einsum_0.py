
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_einsum_inputs():
    list_of_inputs = []

    # Input 1: Transpose of a 2D matrix
    input_dict = {
        'subscripts': 'ij->ji',
        'operands': np.arange(6, dtype=np.float32).reshape(2, 3)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Sum over all axes to a scalar
    input_dict = {
        'subscripts': 'ij->',
        'operands': np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Sum over a single axis
    input_dict = {
        'subscripts': 'ijk->ik',
        'operands': np.arange(-12, 12, dtype=np.float64).reshape(2, 3, 4)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Get diagonal of a square matrix
    input_dict = {
        'subscripts': 'ii->i',
        'operands': np.arange(25, dtype=np.int64).reshape(5, 5)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Trace of a square matrix (sum of diagonal)
    input_dict = {
        'subscripts': 'ii->',
        'operands': np.eye(4, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Permute axes of a 3D tensor
    input_dict = {
        'subscripts': 'ijk->kji',
        'operands': np.random.rand(2, 3, 4).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Sum over one axis using ellipsis
    input_dict = {
        'subscripts': '...k->...',
        'operands': np.ones((5, 2, 3), dtype=np.float16)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Sum all elements of a tensor using ellipsis
    input_dict = {
        'subscripts': '...->',
        'operands': np.arange(24).reshape(2, 3, 4)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Identity operation on a 4D tensor
    input_dict = {
        'subscripts': 'abcd->abcd',
        'operands': np.random.uniform(size=(1, 2, 2, 1)).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Sum over two axes of a 4D tensor
    input_dict = {
        'subscripts': 'abcd->ad',
        'operands': np.arange(24, dtype=np.int32).reshape(1, 2, 3, 4)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Identity operation using ellipsis
    input_dict = {
        'subscripts': '...->...',
        'operands': np.arange(1, dtype=np.int32).reshape(1, 1, 1, 1, 1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: Permute last two axes using ellipsis
    input_dict = {
        'subscripts': '...ij->...ji',
        'operands': np.arange(24).reshape(2, 3, 4)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.einsum"] = tf_experimental_numpy_einsum_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.einsum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.einsum'.")

check_valid('tf.experimental.numpy.einsum', generated_inputs['tf.experimental.numpy.einsum'], lib="tf", suffix=0)
