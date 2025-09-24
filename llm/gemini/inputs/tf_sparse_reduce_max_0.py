
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_reduce_max_inputs():
    list_of_inputs = []

    class PatchedSparseTensor(tf.SparseTensor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.size = np.prod(self.dense_shape)

    # Input 1: Basic 2D reduction along axis 0
    input_dict = {
        'sp_input': PatchedSparseTensor(
            indices=np.array([[0, 0], [0, 2], [1, 1]], dtype=np.int64),
            values=np.array([1, 2, 3], dtype=np.int32),
            dense_shape=np.array([2, 3], dtype=np.int64)
        ),
        'axis': [0],
        'keepdims': False,
        'output_is_sparse': False,
        'name': 'reduce_max_ax0'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic 2D reduction along axis 1
    input_dict = {
        'sp_input': PatchedSparseTensor(
            indices=np.array([[0, 0], [0, 2], [1, 1]], dtype=np.int64),
            values=np.array([1, 2, 3], dtype=np.int32),
            dense_shape=np.array([2, 3], dtype=np.int64)
        ),
        'axis': [1],
        'keepdims': False,
        'output_is_sparse': False,
        'name': 'reduce_max_ax1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Reduce all axes
    input_dict = {
        'sp_input': PatchedSparseTensor(
            indices=np.array([[0, 0], [0, 2], [1, 1]], dtype=np.int64),
            values=np.array([1, 2, 3], dtype=np.int32),
            dense_shape=np.array([2, 3], dtype=np.int64)
        ),
        'axis': [0, 1],
        'keepdims': False,
        'output_is_sparse': False,
        'name': 'reduce_max_all_axes'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: keepdims=True
    input_dict = {
        'sp_input': PatchedSparseTensor(
            indices=np.array([[0, 0], [0, 2], [1, 1]], dtype=np.int64),
            values=np.array([1, 2, 3], dtype=np.int32),
            dense_shape=np.array([2, 3], dtype=np.int64)
        ),
        'axis': [1],
        'keepdims': True,
        'output_is_sparse': False,
        'name': 'reduce_max_keepdims'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative values
    input_dict = {
        'sp_input': PatchedSparseTensor(
            indices=np.array([[0, 0], [1, 0], [1, 1]], dtype=np.int64),
            values=np.array([-7, -4, -3], dtype=np.int32),
            dense_shape=np.array([3, 2], dtype=np.int64)
        ),
        'axis': [1],
        'keepdims': False,
        'output_is_sparse': False,
        'name': 'reduce_max_negative_vals'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: output_is_sparse=True
    input_dict = {
        'sp_input': PatchedSparseTensor(
            indices=np.array([[0, 0], [0, 2], [1, 1]], dtype=np.int64),
            values=np.array([1, 2, 3], dtype=np.int32),
            dense_shape=np.array([2, 3], dtype=np.int64)
        ),
        'axis': [0],
        'keepdims': False,
        'output_is_sparse': True,
        'name': 'reduce_max_sparse_output'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D tensor, reduce along axis 1
    input_dict = {
        'sp_input': PatchedSparseTensor(
            indices=np.array([[0, 0, 1], [0, 2, 0], [1, 1, 1], [1, 2, 0]], dtype=np.int64),
            values=np.array([5, 8, 2, 9], dtype=np.int32),
            dense_shape=np.array([2, 3, 2], dtype=np.int64)
        ),
        'axis': [1],
        'keepdims': False,
        'output_is_sparse': False,
        'name': 'reduce_max_3d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D tensor, reduce multiple axes
    input_dict = {
        'sp_input': PatchedSparseTensor(
            indices=np.array([[0, 0, 1], [0, 2, 0], [1, 1, 1], [1, 2, 0]], dtype=np.int64),
            values=np.array([5, 8, 2, 9], dtype=np.int32),
            dense_shape=np.array([2, 3, 2], dtype=np.int64)
        ),
        'axis': [0, 2],
        'keepdims': False,
        'output_is_sparse': False,
        'name': 'reduce_max_3d_multi_axis'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Floating point values
    input_dict = {
        'sp_input': PatchedSparseTensor(
            indices=np.array([[0, 1], [1, 0], [1, 2]], dtype=np.int64),
            values=np.array([1.1, -2.2, 3.3], dtype=np.float32),
            dense_shape=np.array([2, 4], dtype=np.int64)
        ),
        'axis': [0],
        'keepdims': False,
        'output_is_sparse': False,
        'name': 'reduce_max_float'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Empty sparse tensor (should reduce to 0)
    input_dict = {
        'sp_input': PatchedSparseTensor(
            indices=np.empty((0, 2), dtype=np.int64),
            values=np.array([], dtype=np.int32),
            dense_shape=np.array([3, 4], dtype=np.int64)
        ),
        'axis': [1],
        'keepdims': False,
        'output_is_sparse': False,
        'name': 'reduce_max_empty'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: Negative axis
    input_dict = {
        'sp_input': PatchedSparseTensor(
            indices=np.array([[0, 0], [0, 2], [1, 1]], dtype=np.int64),
            values=np.array([1, 2, 3], dtype=np.int32),
            dense_shape=np.array([2, 3], dtype=np.int64)
        ),
        'axis': [-1],
        'keepdims': False,
        'output_is_sparse': False,
        'name': 'reduce_max_negative_axis'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: Zero reduction case from docs
    input_dict = {
        'sp_input': PatchedSparseTensor(
            indices=np.array([[0, 0,], [1, 0], [1, 1]], dtype=np.int64),
            values=np.array([-7, 4, 3], dtype=np.int32),
            dense_shape=np.array([3, 2], dtype=np.int64)
        ),
        'axis': [1],
        'keepdims': False,
        'output_is_sparse': False,
        'name': 'reduce_max_zero_reduction'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.sparse.reduce_max"] = tf_sparse_reduce_max_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.sparse.reduce_max' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.reduce_max'.")

check_valid('tf.sparse.reduce_max', generated_inputs['tf.sparse.reduce_max'], lib="tf", suffix=0)
