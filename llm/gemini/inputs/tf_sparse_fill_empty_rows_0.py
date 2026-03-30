
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_fill_empty_rows_inputs():
    list_of_inputs = []

    # Input 1 - int32
    indices = np.array([[0, 1], [0, 3], [2, 0], [3, 1]], dtype=np.int64)
    values = np.array([1, 2, 3, 4], dtype=np.int32)
    shape = np.array([5, 6], dtype=np.int64)
    sp_input = tf.SparseTensor(indices, values, shape)
    default_value = tf.constant(0, dtype=np.int32)
    name = "test_name_1"
    input_dict = {"sp_input": sp_input, "default_value": default_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2 - float32
    indices = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.int64)
    values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    shape = np.array([3, 3], dtype=np.int64)
    sp_input = tf.SparseTensor(indices, values, shape)
    default_value = tf.constant(0.0, dtype=np.float32)
    name = "test_name_2"
    input_dict = {"sp_input": sp_input, "default_value": default_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3 - empty sparse tensor - int32
    indices = np.array([], dtype=np.int64).reshape(0, 2)
    values = np.array([], dtype=np.int32)
    shape = np.array([5, 5], dtype=np.int64)
    sp_input = tf.SparseTensor(indices, values, shape)
    default_value = tf.constant(5, dtype=np.int32)
    name = "test_name_3"
    input_dict = {"sp_input": sp_input, "default_value": default_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4 - all rows filled - float32
    indices = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]], dtype=np.int64)
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    shape = np.array([5, 5], dtype=np.int64)
    sp_input = tf.SparseTensor(indices, values, shape)
    default_value = tf.constant(-1.0, dtype=np.float32)
    name = "test_name_4"
    input_dict = {"sp_input": sp_input, "default_value": default_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5 - float64
    indices = np.array([[0, 0], [2, 1]], dtype=np.int64)
    values = np.array([1.0, 2.0], dtype=np.float64)
    shape = np.array([4, 2], dtype=np.int64)
    sp_input = tf.SparseTensor(indices, values, shape)
    default_value = tf.constant(0.0, dtype=np.float64)
    name = "test_name_5"
    input_dict = {"sp_input": sp_input, "default_value": default_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6 - int64
    indices = np.array([[0, 0], [2, 1]], dtype=np.int64)
    values = np.array([1, 2], dtype=np.int64)
    shape = np.array([4, 2], dtype=np.int64)
    sp_input = tf.SparseTensor(indices, values, shape)
    default_value = tf.constant(0, dtype=np.int64)
    name = "test_name_6"
    input_dict = {"sp_input": sp_input, "default_value": default_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7 - Empty rows at the beginning - int32
    indices = np.array([[2, 0], [3, 1]], dtype=np.int64)
    values = np.array([3, 4], dtype=np.int32)
    shape = np.array([5, 6], dtype=np.int64)
    sp_input = tf.SparseTensor(indices, values, shape)
    default_value = tf.constant(0, dtype=np.int32)
    name = "test_name_7"
    input_dict = {"sp_input": sp_input, "default_value": default_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8 - boolean
    indices = np.array([[0, 0], [2, 1]], dtype=np.int64)
    values = np.array([True, False], dtype=np.bool_)
    shape = np.array([4, 2], dtype=np.int64)
    sp_input = tf.SparseTensor(indices, values, shape)
    default_value = tf.constant(False, dtype=np.bool_)
    name = "test_name_8"
    input_dict = {"sp_input": sp_input, "default_value": default_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 - Small Shape - int32
    indices = np.array([[0, 0]], dtype=np.int64)
    values = np.array([1], dtype=np.int32)
    shape = np.array([1, 1], dtype=np.int64)
    sp_input = tf.SparseTensor(indices, values, shape)
    default_value = tf.constant(0, dtype=np.int32)
    name = "test_name_9"
    input_dict = {"sp_input": sp_input, "default_value": default_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 - Large Shape and all zeros - float64
    indices = np.array([[99, 99]], dtype=np.int64)
    values = np.array([1.0], dtype=np.float64)
    shape = np.array([100, 100], dtype=np.int64)
    sp_input = tf.SparseTensor(indices, values, shape)
    default_value = tf.constant(0.0, dtype=np.float64)
    name = "test_name_10"
    input_dict = {"sp_input": sp_input, "default_value": default_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11 - Different indices - int32
    indices = np.array([[1, 0], [3, 2]], dtype=np.int64)
    values = np.array([5, 6], dtype=np.int32)
    shape = np.array([5, 6], dtype=np.int64)
    sp_input = tf.SparseTensor(indices, values, shape)
    default_value = tf.constant(0, dtype=np.int32)
    name = "test_name_11"
    input_dict = {"sp_input": sp_input, "default_value": default_value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.sparse.fill_empty_rows"] = tf_sparse_fill_empty_rows_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.sparse.fill_empty_rows' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.fill_empty_rows'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.sparse.fill_empty_rows', generated_inputs['tf.sparse.fill_empty_rows'], lib="tf", suffix=0)
