
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def get_tf_edit_distance_inputs():
    """
    Generates a list of valid inputs for the tf.edit_distance function.
    The 'hypothesis' and 'truth' sparse tensors are represented as tuples
    of (indices, values, dense_shape) numpy arrays, from which the
    test harness is expected to construct tf.SparseTensor objects.
    """
    list_of_inputs = []

    def create_sparse_tuple(indices, values, shape, value_dtype=np.int64):
        """
        Creates a tuple of numpy arrays representing a sparse tensor.
        (indices, values, dense_shape)
        """
        rank = len(shape)
        if not indices:
            indices_np = np.empty((0, rank), dtype=np.int64)
        else:
            indices_np = np.array(indices, dtype=np.int64)

        values_np = np.array(values, dtype=value_dtype)
        shape_np = np.array(shape, dtype=np.int64)

        return (indices_np, values_np, shape_np)

    # The API expects tf.SparseTensor. The test harness seems to fail when
    # given SparseTensor objects directly. A common pattern is to represent
    # them as a tuple of numpy arrays (indices, values, dense_shape)
    # which the harness then uses to construct the actual SparseTensor.

    # Input 1: Basic case, rank 2, identical sequences
    input_dict_1 = {
        'hypothesis': create_sparse_tuple([[0, 0], [0, 1], [0, 2]], [97, 98, 99], [1, 3]),
        'truth': create_sparse_tuple([[0, 0], [0, 1], [0, 2]], [97, 98, 99], [1, 3]),
        'normalize': False,
        'name': 'identical_seqs'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Basic case, rank 2, one substitution, normalize=False
    input_dict_2 = {
        'hypothesis': create_sparse_tuple([[0, 0], [0, 1], [0, 2]], [99, 97, 116], [1, 3]),
        'truth': create_sparse_tuple([[0, 0], [0, 1], [0, 2]], [99, 117, 116], [1, 3]),
        'normalize': False,
        'name': 'one_substitution'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: One deletion, normalized=True
    input_dict_3 = {
        'hypothesis': create_sparse_tuple([[0, 0], [0, 1]], [97, 99], [1, 3]),
        'truth': create_sparse_tuple([[0, 0], [0, 1], [0, 2]], [97, 98, 99], [1, 3]),
        'normalize': True,
        'name': 'one_deletion_normalized'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: One insertion, normalized=True
    input_dict_4 = {
        'hypothesis': create_sparse_tuple([[0, 0], [0, 1], [0, 2], [0, 3]], [97, 98, 120, 99], [1, 4]),
        'truth': create_sparse_tuple([[0, 0], [0, 1], [0, 2]], [97, 98, 99], [1, 4]),
        'normalize': True,
        'name': 'one_insertion_normalized'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Empty hypothesis sequence
    input_dict_5 = {
        'hypothesis': create_sparse_tuple([], [], [1, 5]),
        'truth': create_sparse_tuple([[0, 0], [0, 1], [0, 2], [0, 3]], [119, 111, 114, 100], [1, 5]),
        'normalize': False,
        'name': 'empty_hypothesis'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Empty truth sequence, normalized
    input_dict_6 = {
        'hypothesis': create_sparse_tuple([[0, 0], [0, 1], [0, 2], [0, 3]], [119, 111, 114, 100], [1, 5]),
        'truth': create_sparse_tuple([], [], [1, 5]),
        'normalize': True,
        'name': 'empty_truth_normalized'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Batch of 2 sequences, rank 2
    input_dict_7 = {
        'hypothesis': create_sparse_tuple([[0, 0], [1, 0], [1, 1]], [97, 98, 99], [2, 3]),
        'truth': create_sparse_tuple([[0, 0], [0, 1], [1, 0], [1, 1]], [97, 120, 98, 99], [2, 3]),
        'normalize': False,
        'name': 'batch_of_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Rank 3 input, matching shapes
    input_dict_8 = {
        'hypothesis': create_sparse_tuple([[0, 0, 0], [0, 1, 0]], [104, 105], [1, 2, 2]),
        'truth': create_sparse_tuple([[0, 0, 0], [0, 1, 1]], [104, 111], [1, 2, 2]),
        'normalize': False,
        'name': 'rank_3_input'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Using float values
    input_dict_9 = {
        'hypothesis': create_sparse_tuple([[0, 0], [0, 1], [0, 2]], [1.0, 2.0, 3.0], [1, 4], value_dtype=np.float32),
        'truth': create_sparse_tuple([[0, 0], [0, 1], [0, 2]], [1.0, 9.0, 3.0], [1, 4], value_dtype=np.float32),
        'normalize': False,
        'name': 'float_values'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Completely different sequences, normalized
    input_dict_10 = {
        'hypothesis': create_sparse_tuple([[0, 0], [0, 1], [0, 2]], [97, 98, 99], [1, 3]),
        'truth': create_sparse_tuple([[0, 0], [0, 1], [0, 2]], [120, 121, 122], [1, 3]),
        'normalize': True,
        'name': 'different_seqs_normalized'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.edit_distance"] = get_tf_edit_distance_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.edit_distance' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.edit_distance'.")

check_valid('tf.edit_distance', generated_inputs['tf.edit_distance'], lib="tf", suffix=0)
