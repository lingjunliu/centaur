
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_experimental_numpy_einsum_inputs():
    list_of_inputs = []

    # Input 1: Transpose a matrix
    input_dict = {
        'subscripts': 'ij->ji',
        'operands': np.arange(6).reshape(2, 3).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Sum all elements of a tensor
    input_dict = {
        'subscripts': '...->',
        'operands': np.arange(24).reshape(2, 3, 4).astype(np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Get the diagonal of a matrix
    input_dict = {
        'subscripts': 'ii->i',
        'operands': np.arange(9).reshape(3, 3).astype(np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Calculate the trace of a matrix
    input_dict = {
        'subscripts': 'ii->',
        'operands': np.array([[-1, 2, 3], [4, -5, 6], [7, 8, -9]], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Sum along a specific axis
    input_dict = {
        'subscripts': 'ijk->ik',
        'operands': np.arange(60).reshape(3, 4, 5).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Permute tensor dimensions
    input_dict = {
        'subscripts': 'ijk->kji',
        'operands': np.random.rand(2, 3, 4).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Identity operation
    input_dict = {
        'subscripts': 'abc->abc',
        'operands': np.random.rand(5, 1, 3).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Create a diagonal matrix from a vector
    input_dict = {
        'subscripts': 'i->ii',
        'operands': np.arange(5).astype(np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Sum along the first axis
    input_dict = {
        'subscripts': 'i...->...',
        'operands': np.arange(12).reshape(3, 2, 2).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Sum along the last axis
    input_dict = {
        'subscripts': '...i->...',
        'operands': np.arange(12).reshape(2, 2, 3).astype(np.float32)
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
