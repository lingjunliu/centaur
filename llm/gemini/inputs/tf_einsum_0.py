
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy


def tf_einsum_inputs():
    """
    Generates a list of valid inputs for the tf.einsum function.
    """
    list_of_inputs = []

    # The test harness appears to pass the value of the 'inputs' key as a
    # single argument to tf.einsum. Therefore, only single-tensor operations
    # are generated here. Multi-tensor equations like 'ij,jk->ik' would fail
    # as TensorFlow would receive 1 input tensor but expect 2.

    # Input 1: Transpose a matrix
    input_1 = {
        'equation': 'ij->ji',
        'inputs': np.random.rand(3, 5).astype(np.float32),
        'optimize': 'greedy',
        'name': 'transpose'
    }
    list_of_inputs.append(copy.deepcopy(input_1))

    # Input 2: Calculate the trace of a matrix
    input_2 = {
        'equation': 'ii',
        'inputs': np.random.rand(4, 4).astype(np.float32),
        'optimize': 'greedy',
        'name': 'trace'
    }
    list_of_inputs.append(copy.deepcopy(input_2))

    # Input 3: Extract the diagonal of a matrix
    input_3 = {
        'equation': 'ii->i',
        'inputs': np.arange(25, dtype=np.int32).reshape(5, 5),
        'optimize': 'greedy',
        'name': 'diagonal'
    }
    list_of_inputs.append(copy.deepcopy(input_3))

    # Input 4: Sum all elements of a matrix to a scalar
    input_4 = {
        'equation': 'ij->',
        'inputs': np.ones((3, 4), dtype=np.float64),
        'optimize': 'greedy',
        'name': 'sum_all'
    }
    list_of_inputs.append(copy.deepcopy(input_4))

    # Input 5: Sum a tensor along a specific axis
    input_5 = {
        'equation': 'ijk->ik',
        'inputs': np.random.rand(2, 3, 4).astype(np.float32),
        'optimize': 'greedy',
        'name': 'sum_axis'
    }
    list_of_inputs.append(copy.deepcopy(input_5))

    # Input 6: Permute axes of a 4D tensor
    input_6 = {
        'equation': 'abcd->badc',
        'inputs': np.random.rand(2, 3, 4, 5).astype(np.float32),
        'optimize': 'greedy',
        'name': 'permute_axes'
    }
    list_of_inputs.append(copy.deepcopy(input_6))

    # Input 7: Batch transpose using ellipsis
    input_7 = {
        'equation': '...ij->...ji',
        'inputs': np.random.rand(10, 3, 5).astype(np.float32),
        'optimize': 'auto',
        'name': 'batch_transpose'
    }
    list_of_inputs.append(copy.deepcopy(input_7))

    # Input 8: Batch diagonal extraction using ellipsis
    input_8 = {
        'equation': '...ii->...i',
        'inputs': np.random.rand(5, 4, 4).astype(np.float32),
        'optimize': 'optimal',
        'name': 'batch_diagonal'
    }
    list_of_inputs.append(copy.deepcopy(input_8))

    # Input 9: Sum a matrix along the first axis
    input_9 = {
        'equation': 'ij->j',
        'inputs': np.random.rand(7, 8).astype(np.float32),
        'optimize': 'greedy',
        'name': 'sum_axis_2'
    }
    list_of_inputs.append(copy.deepcopy(input_9))

    # Input 10: Another axis permutation
    input_10 = {
        'equation': 'ijk->kji',
        'inputs': np.random.rand(2, 3, 4).astype(np.float32),
        'optimize': 'greedy',
        'name': 'permute_axes_2'
    }
    list_of_inputs.append(copy.deepcopy(input_10))

    return list_of_inputs

generated_inputs["tf.einsum"] = tf_einsum_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.einsum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.einsum'.")

check_valid('tf.einsum', generated_inputs['tf.einsum'], lib="tf", suffix=0)
