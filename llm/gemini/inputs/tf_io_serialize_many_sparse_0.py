
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_serialize_many_sparse_inputs():
    list_of_inputs = []

    def create_dense_from_sparse(indices, values, shape):
        dtype = values.dtype
        
        # Initialize dense array based on dtype
        if dtype == object:
            # np.full doesn't work well with object arrays and empty strings
            dense = np.empty(shape, dtype=object)
            dense.fill('')
        else:
            fill_value = 0
            if np.issubdtype(dtype, np.complexfloating):
                fill_value = 0j
            elif np.issubdtype(dtype, np.bool_):
                fill_value = False
            
            # For shapes with a dimension of 0, np.full can handle it.
            dense = np.full(shape, fill_value, dtype=dtype)
        
        # Populate the dense array with sparse values
        if len(indices) > 0 and np.prod(shape) > 0:
            dense[tuple(indices.T)] = values
            
        return dense

    # Input 1: Basic 2D
    indices = np.array([[0, 1], [0, 3], [1, 2]], dtype=np.int64)
    values = np.array([1, 2, 3], dtype=np.int32)
    dense_shape = (2, 5)
    sp_input = create_dense_from_sparse(indices, values, dense_shape)
    input_dict = {
        'sp_input': sp_input,
        'out_type': tf.string,
        'name': 'basic_2d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3D float values
    indices = np.array([[0, 1, 1], [0, 2, 3], [2, 0, 0], [2, 1, 2]], dtype=np.int64)
    values = np.array([10.5, 20.2, 30.8, 40.1], dtype=np.float32)
    dense_shape = (3, 3, 4)
    sp_input = create_dense_from_sparse(indices, values, dense_shape)
    input_dict = {
        'sp_input': sp_input,
        'out_type': tf.string,
        'name': 'basic_3d_float'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Empty minibatch item
    indices = np.array([[0, 1], [0, 3], [2, 2]], dtype=np.int64)
    values = np.array([-1, -2, -3], dtype=np.int64)
    dense_shape = (3, 5)
    sp_input = create_dense_from_sparse(indices, values, dense_shape)
    input_dict = {
        'sp_input': sp_input,
        'out_type': tf.string,
        'name': 'empty_minibatch_item'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Completely empty Tensor (0 values)
    indices = np.empty((0, 2), dtype=np.int64)
    values = np.array([], dtype=np.float32)
    dense_shape = (3, 4)
    sp_input = create_dense_from_sparse(indices, values, dense_shape)
    input_dict = {
        'sp_input': sp_input,
        'out_type': tf.string,
        'name': 'empty_tensor'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Empty Tensor (0 minibatch size)
    indices = np.empty((0, 3), dtype=np.int64)
    values = np.array([], dtype=np.int32)
    dense_shape = (0, 5, 5)
    sp_input = create_dense_from_sparse(indices, values, dense_shape)
    input_dict = {
        'sp_input': sp_input,
        'out_type': tf.string,
        'name': 'zero_minibatch'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large minibatch size
    indices = np.array([[0, 0, 0], [2, 1, 1], [5, 0, 1], [9, 1, 0]], dtype=np.int64)
    values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    dense_shape = (10, 2, 2)
    sp_input = create_dense_from_sparse(indices, values, dense_shape)
    input_dict = {
        'sp_input': sp_input,
        'out_type': tf.string,
        'name': 'large_minibatch'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Rank 4 Tensor
    indices = np.array([[0, 0, 1, 0], [1, 1, 0, 1]], dtype=np.int64)
    values = np.array([100, 200], dtype=np.int32)
    dense_shape = (2, 2, 2, 2)
    sp_input = create_dense_from_sparse(indices, values, dense_shape)
    input_dict = {
        'sp_input': sp_input,
        'out_type': tf.string,
        'name': 'rank_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: String values
    indices = np.array([[0, 1], [1, 0]], dtype=np.int64)
    values = np.array(['hello', 'world'], dtype=object)
    dense_shape = (2, 2)
    sp_input = create_dense_from_sparse(indices, values, dense_shape)
    input_dict = {
        'sp_input': sp_input,
        'out_type': tf.string,
        'name': 'string_values'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Boolean values
    indices = np.array([[0, 0, 0], [0, 1, 1]], dtype=np.int64)
    values = np.array([True, False], dtype=np.bool_)
    dense_shape = (1, 2, 2)
    sp_input = create_dense_from_sparse(indices, values, dense_shape)
    input_dict = {
        'sp_input': sp_input,
        'out_type': tf.string,
        'name': 'single_item_minibatch_bool'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: out_type as tf.variant
    indices = np.array([[0, 1], [1, 0], [1, 1]], dtype=np.int64)
    values = np.array([10, 20, 30], dtype=np.int16)
    dense_shape = (2, 2)
    sp_input = create_dense_from_sparse(indices, values, dense_shape)
    input_dict = {
        'sp_input': sp_input,
        'out_type': tf.variant,
        'name': 'variant_out_type'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: Complex values
    indices = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int64)
    values = np.array([1+2j, 3-4j], dtype=np.complex64)
    dense_shape = (2, 2, 2)
    sp_input = create_dense_from_sparse(indices, values, dense_shape)
    input_dict = {
        'sp_input': sp_input,
        'out_type': tf.string,
        'name': 'complex_values'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.io.serialize_many_sparse"] = tf_io_serialize_many_sparse_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.io.serialize_many_sparse' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.serialize_many_sparse'.")

check_valid('tf.io.serialize_many_sparse', generated_inputs['tf.io.serialize_many_sparse'], lib="tf", suffix=0)
