
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

# The testing harness requires tensor-like objects to have a `.size` attribute,
# which is not present on `tf.sparse.SparseTensor`. This helper class adds it.
class DuckTypedSparseTensor(tf.sparse.SparseTensor):
    @property
    def size(self):
        shape = tf.get_static_value(self.dense_shape)
        if shape is None:
            return 0
        return np.prod(shape).item()

def tf_sparse_to_dense_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D integer tensor
    sp_input_1 = DuckTypedSparseTensor(
        indices=np.array([[0, 1], [2, 3]], dtype=np.int64),
        values=np.array([10, 20], dtype=np.int32),
        dense_shape=np.array([3, 4], dtype=np.int64))
    input_dict_1 = {
        'sp_input': sp_input_1,
        'default_value': np.array(0, dtype=np.int32),
        'validate_indices': True,
        'name': 'basic_int_2d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 2D float tensor with a float default value
    sp_input_2 = DuckTypedSparseTensor(
        indices=np.array([[0, 0], [1, 1]], dtype=np.int64),
        values=np.array([-1.5, 2.5], dtype=np.float32),
        dense_shape=np.array([2, 2], dtype=np.int64))
    input_dict_2 = {
        'sp_input': sp_input_2,
        'default_value': np.array(0.1, dtype=np.float32),
        'validate_indices': True,
        'name': 'basic_float_2d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 3D tensor with a negative integer default value
    sp_input_3 = DuckTypedSparseTensor(
        indices=np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int64),
        values=np.array([100, 200], dtype=np.int32),
        dense_shape=np.array([2, 2, 2], dtype=np.int64))
    input_dict_3 = {
        'sp_input': sp_input_3,
        'default_value': np.array(-1, dtype=np.int32),
        'validate_indices': True,
        'name': 'basic_int_3d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: 1D tensor (vector)
    sp_input_4 = DuckTypedSparseTensor(
        indices=np.array([[2], [5]], dtype=np.int64),
        values=np.array([3.0, 6.0], dtype=np.float32),
        dense_shape=np.array([8], dtype=np.int64))
    input_dict_4 = {
        'sp_input': sp_input_4,
        'default_value': np.array(0.0, dtype=np.float32),
        'validate_indices': True,
        'name': 'vector_float'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: `validate_indices` is False with unsorted indices
    sp_input_5 = DuckTypedSparseTensor(
        indices=np.array([[1, 0], [0, 1]], dtype=np.int64),
        values=np.array([5, 6], dtype=np.int32),
        dense_shape=np.array([2, 2], dtype=np.int64))
    input_dict_5 = {
        'sp_input': sp_input_5,
        'default_value': np.array(0, dtype=np.int32),
        'validate_indices': False,
        'name': 'validate_false'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Empty sparse tensor (all values will be default)
    sp_input_6 = DuckTypedSparseTensor(
        indices=np.empty(shape=(0, 2), dtype=np.int64),
        values=np.array([], dtype=np.int32),
        dense_shape=np.array([4, 4], dtype=np.int64))
    input_dict_6 = {
        'sp_input': sp_input_6,
        'default_value': np.array(99, dtype=np.int32),
        'validate_indices': True,
        'name': 'empty_tensor'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Fully specified sparse tensor
    sp_input_7 = DuckTypedSparseTensor(
        indices=np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64),
        values=np.array([1.1, 2.2, 3.3, 4.4], dtype=np.float32),
        dense_shape=np.array([2, 2], dtype=np.int64))
    input_dict_7 = {
        'sp_input': sp_input_7,
        'default_value': np.array(0.0, dtype=np.float32),
        'validate_indices': True,
        'name': 'full_tensor'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: A sparse tensor with a larger shape
    sp_input_8 = DuckTypedSparseTensor(
        indices=np.array([[5, 10], [15, 20]], dtype=np.int64),
        values=np.array([-1, -2], dtype=np.int32),
        dense_shape=np.array([25, 25], dtype=np.int64))
    input_dict_8 = {
        'sp_input': sp_input_8,
        'default_value': np.array(1, dtype=np.int32),
        'validate_indices': True,
        'name': 'large_shape'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Using int64 values and default_value
    sp_input_9 = DuckTypedSparseTensor(
        indices=np.array([[0, 0]], dtype=np.int64),
        values=np.array([1234567890123], dtype=np.int64),
        dense_shape=np.array([1, 1], dtype=np.int64))
    input_dict_9 = {
        'sp_input': sp_input_9,
        'default_value': np.array(0, dtype=np.int64),
        'validate_indices': True,
        'name': 'int64_values'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Using complex values and default_value
    sp_input_10 = DuckTypedSparseTensor(
        indices=np.array([[0, 1], [1, 0]], dtype=np.int64),
        values=np.array([1+2j, 3-4j], dtype=np.complex64),
        dense_shape=np.array([2, 2], dtype=np.int64))
    input_dict_10 = {
        'sp_input': sp_input_10,
        'default_value': np.array(0j, dtype=np.complex64),
        'validate_indices': True,
        'name': 'complex_values'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.sparse.to_dense"] = tf_sparse_to_dense_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.sparse.to_dense' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.to_dense'.")

check_valid('tf.sparse.to_dense', generated_inputs['tf.sparse.to_dense'], lib="tf", suffix=0)
