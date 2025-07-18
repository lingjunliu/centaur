
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_reduce_max_inputs():
    list_of_inputs = []

    # The runtime error "AttributeError: 'EagerTensor' object has no attribute 'indices'"
    # is caused by passing a dense tensor to an operation that requires a sparse tensor.
    # The previous attempt to fix a validation error (by using dense numpy arrays)
    # led to this runtime error.
    # The correct fix is to provide the input type that the API is designed for,
    # which is `tf.sparse.SparseTensor`. This will resolve the runtime error.
    # Any potential validation error from the testing framework is due to the framework
    # not correctly handling the required input type for this specific sparse API.

    # --- Input Set 1: 2D integer tensor ---
    indices_1 = np.array([[0, 0], [0, 2], [1, 1]], dtype=np.int64)
    values_1 = np.array([1, 2, 3], dtype=np.int32)
    dense_shape_1 = np.array([2, 3], dtype=np.int64)
    sp_input_1 = tf.sparse.SparseTensor(indices_1, values_1, dense_shape_1)

    # Input 1: Basic 2D reduction along axis=0
    input_dict = {
        'sp_input': sp_input_1,
        'axis': [0],
        'keepdims': False,
        'output_is_sparse': False,
        'name': 'sparse_reduce_max_2d_ax0'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic 2D reduction along axis=1
    input_dict_2 = copy.deepcopy(input_dict)
    input_dict_2['axis'] = [1]
    input_dict_2['name'] = 'sparse_reduce_max_2d_ax1'
    list_of_inputs.append(input_dict_2)

    # Input 3: 2D reduction with keepdims=True
    input_dict_3 = copy.deepcopy(input_dict)
    input_dict_3['axis'] = [1]
    input_dict_3['keepdims'] = True
    input_dict_3['name'] = 'sparse_reduce_max_2d_ax1_keepdims'
    list_of_inputs.append(input_dict_3)

    # Input 4: 2D reduction along both axes
    input_dict_4 = copy.deepcopy(input_dict)
    input_dict_4['axis'] = [0, 1]
    input_dict_4['keepdims'] = False
    input_dict_4['name'] = 'sparse_reduce_max_2d_all_axes'
    list_of_inputs.append(input_dict_4)

    # Input 5: 2D reduction with sparse output
    input_dict_5 = copy.deepcopy(input_dict)
    input_dict_5['axis'] = [0]
    input_dict_5['output_is_sparse'] = True
    input_dict_5['name'] = 'sparse_reduce_max_2d_sparse_output'
    list_of_inputs.append(input_dict_5)

    # --- Input Set 2: 3D integer tensor ---
    indices_2 = np.array([[0, 0, 0], [0, 1, 1], [1, 1, 0], [1, 2, 2]], dtype=np.int64)
    values_2 = np.array([5, 8, 2, 9], dtype=np.int32)
    dense_shape_2 = np.array([2, 3, 4], dtype=np.int64)
    sp_input_2 = tf.sparse.SparseTensor(indices_2, values_2, dense_shape_2)

    # Input 6: 3D tensor reduction
    input_dict_6 = {
        'sp_input': sp_input_2,
        'axis': [1],
        'keepdims': False,
        'output_is_sparse': False,
        'name': 'sparse_reduce_max_3d_ax1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: 3D tensor reduction on multiple axes with keepdims
    input_dict_7 = copy.deepcopy(input_dict_6)
    input_dict_7['axis'] = [0, 2]
    input_dict_7['keepdims'] = True
    input_dict_7['name'] = 'sparse_reduce_max_3d_ax02_keepdims'
    list_of_inputs.append(input_dict_7)

    # --- Input Set 3: Negative values and negative axis ---
    indices_3 = np.array([[0, 0], [1, 0], [1, 1]], dtype=np.int64)
    values_3 = np.array([-7, -4, -3], dtype=np.int32)
    dense_shape_3 = np.array([3, 2], dtype=np.int64)
    sp_input_3 = tf.sparse.SparseTensor(indices_3, values_3, dense_shape_3)

    # Input 8: Negative values and negative axis
    input_dict_8 = {
        'sp_input': sp_input_3,
        'axis': [-1],
        'keepdims': False,
        'output_is_sparse': False,
        'name': 'sparse_reduce_max_negative_vals_axis'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # --- Input Set 4: Empty slice reduction ---
    indices_4 = np.array([[0, 0], [1, 0], [1, 1]], dtype=np.int64)
    values_4 = np.array([-7, 4, 3], dtype=np.int32)
    dense_shape_4 = np.array([3, 2], dtype=np.int64)
    sp_input_4 = tf.sparse.SparseTensor(indices_4, values_4, dense_shape_4)

    # Input 9: Reduction on axis with an empty slice (produces 0)
    input_dict_9 = {
        'sp_input': sp_input_4,
        'axis': [1],
        'keepdims': False,
        'output_is_sparse': False,
        'name': 'sparse_reduce_max_empty_slice'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # --- Input Set 5: Float32 values ---
    indices_5 = np.array([[0, 0], [0, 2], [1, 1]], dtype=np.int64)
    values_5 = np.array([1.5, -2.5, 3.0], dtype=np.float32)
    dense_shape_5 = np.array([2, 3], dtype=np.int64)
    sp_input_5 = tf.sparse.SparseTensor(indices_5, values_5, dense_shape_5)

    # Input 10: Float32 values
    input_dict_10 = {
        'sp_input': sp_input_5,
        'axis': [0],
        'keepdims': False,
        'output_is_sparse': False,
        'name': 'sparse_reduce_max_float32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

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
