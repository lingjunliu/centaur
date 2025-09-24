
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tfe_numpy_einsum_inputs():
    """
    Generates a list of valid inputs for tf.experimental.numpy.einsum.
    Based on the observed errors, the 'operands' argument is treated as a single
    tensor, not a list of tensors to be unpacked. Therefore, only single-operand
    einsum examples are provided, and the operand tensor is passed directly
    without being wrapped in an outer list or array.
    """
    list_of_inputs = []

    # Input 1: Transpose a 2D matrix
    input_dict_1 = {
        'subscripts': 'ij->ji',
        'operands': np.arange(6).reshape(2, 3)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Sum over all axes
    input_dict_2 = {
        'subscripts': 'ij->',
        'operands': np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Extract the diagonal
    input_dict_3 = {
        'subscripts': 'ii->i',
        'operands': np.arange(9).reshape(3, 3)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Calculate the trace (sum of diagonal)
    input_dict_4 = {
        'subscripts': 'ii',
        'operands': np.array([[-5, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Permute dimensions of a 3D tensor
    input_dict_5 = {
        'subscripts': 'ijk->kji',
        'operands': np.arange(24).reshape(2, 3, 4).astype(np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Sum over the second axis (rows)
    input_dict_6 = {
        'subscripts': 'ij->i',
        'operands': np.arange(12).reshape(3, 4)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Permute dimensions of a 4D tensor
    input_dict_7 = {
        'subscripts': 'abcd->adcb',
        'operands': np.random.randn(2, 3, 4, 5)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Higher-order trace
    input_dict_8 = {
        'subscripts': 'abcc->ab',
        'operands': np.arange(2 * 3 * 4 * 4).reshape(2, 3, 4, 4)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Identity operation using ellipsis
    input_dict_9 = {
        'subscripts': '...',
        'operands': np.arange(60).reshape(3, 4, 5)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Sum over the first axis (columns)
    input_dict_10 = {
        'subscripts': 'ij->j',
        'operands': np.arange(12).reshape(3, 4)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: 1D vector to scalar sum
    input_dict_11 = {
        'subscripts': 'i->',
        'operands': np.array([-1, 0, 1, 2, -3])
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    # Input 12: Sum over last dimension using ellipsis
    input_dict_12 = {
        'subscripts': '...i->...',
        'operands': np.arange(24).reshape(2, 3, 4)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_12))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.einsum"] = tfe_numpy_einsum_inputs()

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
