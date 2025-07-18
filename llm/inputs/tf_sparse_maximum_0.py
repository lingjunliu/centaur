
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_maximum_inputs():
    """
    Generates a list of valid inputs for the tf.sparse.maximum function.
    """
    list_of_inputs = []

    # Helper function to create SparseTensor objects as required by the API.
    def _create_sparse_tensor(indices, values, dense_shape, dtype):
        # For an empty sparse tensor, the indices must have shape (0, rank).
        if len(indices) == 0:
            rank = len(dense_shape)
            indices_np = np.empty((0, rank), dtype=np.int64)
        else:
            indices_np = np.array(indices, dtype=np.int64)
        
        # The API requires SparseTensor objects.
        # All provided inputs have lexicographically ordered indices as per the docs.
        return tf.sparse.SparseTensor(
            indices=indices_np,
            values=np.array(values, dtype=dtype),
            dense_shape=np.array(dense_shape, dtype=np.int64)
        )

    # Input 1: Basic 1D case, non-overlapping indices
    input_dict = {
        'sp_a': _create_sparse_tensor([[0], [2]], [10, 30], [5], np.int32),
        'sp_b': _create_sparse_tensor([[1], [4]], [25, 5], [5], np.int32),
        'name': 'basic_1d_non_overlapping'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D with overlapping indices and negative values
    input_dict = {
        'sp_a': _create_sparse_tensor([[0], [2], [4]], [10, -30, 50], [5], np.int32),
        'sp_b': _create_sparse_tensor([[0], [3], [4]], [20, 40, -50], [5], np.int32),
        'name': '1d_overlap_negatives'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Basic 2D case, non-overlapping
    input_dict = {
        'sp_a': _create_sparse_tensor([[0, 0], [1, 1]], [1, 4], [2, 2], np.int32),
        'sp_b': _create_sparse_tensor([[0, 1], [1, 0]], [2, 3], [2, 2], np.int32),
        'name': 'basic_2d_non_overlapping'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D with overlapping indices and float values
    input_dict = {
        'sp_a': _create_sparse_tensor([[0, 1], [1, 2], [2, 0]], [1.5, -2.5, 3.5], [3, 3], np.float32),
        'sp_b': _create_sparse_tensor([[0, 1], [2, 0], [2, 2]], [-1.0, 4.0, 5.0], [3, 3], np.float32),
        'name': '2d_overlap_floats'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: One tensor is empty
    input_dict = {
        'sp_a': _create_sparse_tensor([[0, 0], [1, 1]], [10, 20], [2, 2], np.int64),
        'sp_b': _create_sparse_tensor([], [], [2, 2], np.int64),
        'name': 'one_empty'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Both tensors are empty
    input_dict = {
        'sp_a': _create_sparse_tensor([], [], [4, 5], np.float64),
        'sp_b': _create_sparse_tensor([], [], [4, 5], np.float64),
        'name': 'both_empty'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Basic 3D case
    input_dict = {
        'sp_a': _create_sparse_tensor([[0, 1, 0], [1, 0, 1]], [5, 6], [2, 2, 2], np.int32),
        'sp_b': _create_sparse_tensor([[0, 0, 0], [1, 1, 1]], [7, 8], [2, 2, 2], np.int32),
        'name': 'basic_3d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: All values in one tensor are smaller
    input_dict = {
        'sp_a': _create_sparse_tensor([[0], [2]], [-10, -20], [4], np.int32),
        'sp_b': _create_sparse_tensor([[0], [1]], [-5, 5], [4], np.int32),
        'name': 'a_smaller_than_b'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Identical sparse tensors
    input_dict = {
        'sp_a': _create_sparse_tensor([[0, 1], [2, 0]], [1.0, 2.0], [3, 3], np.float64),
        'sp_b': _create_sparse_tensor([[0, 1], [2, 0]], [1.0, 2.0], [3, 3], np.float64),
        'name': 'identical_tensors'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: One tensor has only negative values, the other has only positive values
    input_dict = {
        'sp_a': _create_sparse_tensor([[0], [2]], [-1, -2], [5], np.int32),
        'sp_b': _create_sparse_tensor([[1], [3]], [3, 4], [5], np.int32),
        'name': 'neg_vs_pos'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.sparse.maximum"] = tf_sparse_maximum_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.sparse.maximum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.maximum'.")

check_valid('tf.sparse.maximum', generated_inputs['tf.sparse.maximum'], lib="tf", suffix=0)
