
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_split_inputs():
    list_of_inputs = []

    # Helper to create a reordered SparseTensor
    def create_reordered_sparse_tensor(indices, values, dense_shape):
        st = tf.sparse.SparseTensor(
            indices=np.array(indices, dtype=np.int64),
            values=values,
            dense_shape=np.array(dense_shape, dtype=np.int64)
        )
        return tf.sparse.reorder(st)

    # Input 1: Basic case from documentation (2D, axis=1, not evenly divisible)
    sp_input_1 = create_reordered_sparse_tensor(
        indices=[[0, 2], [0, 4], [0, 5], [1, 0], [1, 1]],
        values=np.array([1, 2, 3, 4, 5], dtype=np.int32),
        dense_shape=[2, 7]
    )
    input_dict_1 = {
        'sp_input': sp_input_1,
        'num_split': 2,
        'axis': np.array(1, dtype=np.int32),
        'name': 'split_doc_example_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Basic case from documentation (2D, axis=0, evenly divisible)
    input_dict_2 = {
        'sp_input': sp_input_1,
        'num_split': 2,
        'axis': np.array(0, dtype=np.int32),
        'name': 'split_doc_example_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Negative axis
    input_dict_3 = {
        'sp_input': sp_input_1,
        'num_split': 2,
        'axis': np.array(-1, dtype=np.int32),
        'name': 'split_negative_axis'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Evenly divisible on axis=1
    sp_input_4 = create_reordered_sparse_tensor(
        indices=[[0, 1], [0, 5], [1, 2], [1, 7]],
        values=np.array([10, 20, 30, 40], dtype=np.int32),
        dense_shape=[2, 8]
    )
    input_dict_4 = {
        'sp_input': sp_input_4,
        'num_split': 4,
        'axis': np.array(1, dtype=np.int32),
        'name': 'split_even_division'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Higher rank (3D)
    sp_input_5 = create_reordered_sparse_tensor(
        indices=[[0, 0, 1], [0, 1, 2], [1, 2, 3], [1, 3, 0]],
        values=np.array([1.1, 2.2, 3.3, 4.4], dtype=np.float32),
        dense_shape=[2, 4, 5]
    )
    input_dict_5 = {
        'sp_input': sp_input_5,
        'num_split': 2,
        'axis': np.array(1, dtype=np.int32),
        'name': 'split_3d_tensor'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Different DType (float64) and negative values
    sp_input_6 = create_reordered_sparse_tensor(
        indices=[[0, 2], [1, 1]],
        values=np.array([-1.5, 2.5], dtype=np.float64),
        dense_shape=[2, 4]
    )
    input_dict_6 = {
        'sp_input': sp_input_6,
        'num_split': 2,
        'axis': np.array(1, dtype=np.int32),
        'name': 'split_float_neg_values'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Empty SparseTensor
    sp_input_7 = tf.sparse.SparseTensor(
        indices=np.empty((0, 2), dtype=np.int64),
        values=np.array([], dtype=np.int32),
        dense_shape=np.array([4, 6], dtype=np.int64)
    )
    input_dict_7 = {
        'sp_input': sp_input_7,
        'num_split': 3,
        'axis': np.array(1, dtype=np.int32),
        'name': 'split_empty_tensor'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Split results in empty tensors
    sp_input_8 = create_reordered_sparse_tensor(
        indices=[[0, 0], [0, 1]],
        values=np.array([1, 2], dtype=np.int32),
        dense_shape=[2, 10]
    )
    input_dict_8 = {
        'sp_input': sp_input_8,
        'num_split': 5,
        'axis': np.array(1, dtype=np.int32),
        'name': 'split_with_empty_results'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: 3D tensor, split along axis=2, not evenly divisible
    input_dict_9 = {
        'sp_input': sp_input_5,
        'num_split': 3,
        'axis': np.array(2, dtype=np.int32),
        'name': 'split_3d_axis2_uneven'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: 4D tensor
    sp_input_10 = create_reordered_sparse_tensor(
        indices=[[0, 0, 1, 1], [0, 1, 1, 0], [1, 1, 0, 0]],
        values=np.array([10, 20, 30], dtype=np.int64),
        dense_shape=[2, 2, 2, 2]
    )
    input_dict_10 = {
        'sp_input': sp_input_10,
        'num_split': 2,
        'axis': np.array(-1, dtype=np.int32),
        'name': 'split_4d_tensor_neg_axis'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: Split into 1 part (identity operation)
    input_dict_11 = {
        'sp_input': sp_input_1,
        'num_split': 1,
        'axis': np.array(1, dtype=np.int32),
        'name': 'split_into_one'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    return list_of_inputs

generated_inputs["tf.sparse.split"] = tf_sparse_split_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.sparse.split' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.split'.")

check_valid('tf.sparse.split', generated_inputs['tf.sparse.split'], lib="tf", suffix=0)
