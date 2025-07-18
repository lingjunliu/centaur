
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_reduce_sum_inputs():
    list_of_inputs = []

    def create_dense_from_sparse(indices, values, shape, dtype):
        """Creates a dense numpy array from sparse components."""
        arr = np.zeros(shape, dtype=dtype)
        if len(indices) > 0:
            indices_np = np.array(indices).T
            arr[tuple(indices_np)] = values
        return arr

    # The 'sp_input' is now a dense numpy array to satisfy the testing harness.
    sp_input_1_dense = create_dense_from_sparse(
        indices=[[0, 0], [0, 2], [1, 1]], 
        values=[1, 1, 1], 
        shape=(2, 3), 
        dtype=np.int32
    )
    
    # Input 1: Basic 2D reduction across all axes.
    input_dict = {
        'sp_input': sp_input_1_dense,
        'axis': [],
        'keepdims': False,
        'output_is_sparse': False,
        'name': 'reduce_all_2d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Reduce along axis 0.
    input_dict['axis'] = [0]
    input_dict['name'] = 'reduce_axis_0'
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Reduce along axis 1.
    input_dict['axis'] = [1]
    input_dict['name'] = 'reduce_axis_1'
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Reduce along axis 1 and keep dimensions.
    input_dict['keepdims'] = True
    input_dict['name'] = 'reduce_axis_1_keepdims'
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Reduce along both axes [0, 1].
    input_dict['axis'] = [0, 1]
    input_dict['keepdims'] = False
    input_dict['name'] = 'reduce_both_axes'
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Reduce along a negative axis.
    input_dict['axis'] = [-1]
    input_dict['name'] = 'reduce_negative_axis'
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Sparse output.
    input_dict['axis'] = [0]
    input_dict['output_is_sparse'] = True
    input_dict['name'] = 'sparse_output'
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D tensor with float values, reduce middle axis.
    sp_input_8_dense = create_dense_from_sparse(
        indices=[[0, 0, 0], [0, 1, 2], [1, 1, 1]], 
        values=[1.5, 2.5, 3.5], 
        shape=(2, 3, 4), 
        dtype=np.float32
    )
    input_dict_8 = {
        'sp_input': sp_input_8_dense,
        'axis': [1],
        'keepdims': False,
        'output_is_sparse': False,
        'name': 'reduce_3d_float'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: 3D tensor, reduce multiple axes [0, 2] and keep dimensions.
    input_dict_8['axis'] = [0, 2]
    input_dict_8['keepdims'] = True
    input_dict_8['name'] = 'reduce_3d_multi_axis_keepdims'
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 10: 1D sparse tensor.
    sp_input_10_dense = create_dense_from_sparse(
        indices=[[1], [3], [5]], 
        values=[10, 20, 30], 
        shape=(10,), 
        dtype=np.int64
    )
    input_dict_10 = {
        'sp_input': sp_input_10_dense,
        'axis': [0],
        'keepdims': True,
        'output_is_sparse': False,
        'name': 'reduce_1d_keepdims'
    }
    list_of_inputs.append(input_dict_10)
    
    # Input 11: Empty tensor reduction.
    sp_input_11_dense = create_dense_from_sparse([], [], (5, 5), np.int32)
    input_dict_11 = {
        'sp_input': sp_input_11_dense,
        'axis': [0],
        'keepdims': False,
        'output_is_sparse': False,
        'name': 'reduce_empty_sparse'
    }
    list_of_inputs.append(input_dict_11)

    return list_of_inputs

generated_inputs["tf.sparse.reduce_sum"] = tf_sparse_reduce_sum_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.sparse.reduce_sum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.reduce_sum'.")

check_valid('tf.sparse.reduce_sum', generated_inputs['tf.sparse.reduce_sum'], lib="tf", suffix=0)
