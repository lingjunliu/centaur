
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_minimum_inputs():
    list_of_inputs = []

    # Input 1
    indices_a = np.array([[0, 0], [1, 2]])
    values_a = np.array([1, 2], dtype=np.int32)
    dense_shape_a = np.array([3, 3])
    sp_a = tf.sparse.SparseTensor(indices_a, values_a, dense_shape_a)

    indices_b = np.array([[0, 0], [1, 1]])
    values_b = np.array([3, 4], dtype=np.int32)
    dense_shape_b = np.array([3, 3])
    sp_b = tf.sparse.SparseTensor(indices_b, values_b, dense_shape_b)

    input_dict = {"sp_a": sp_a, "sp_b": sp_b, "name": "minimum_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    indices_a = np.array([[0], [2]])
    values_a = np.array([-1, 5], dtype=np.int32)
    dense_shape_a = np.array([5])
    sp_a = tf.sparse.SparseTensor(indices_a, values_a, dense_shape_a)

    indices_b = np.array([[1], [2]])
    values_b = np.array([2, 3], dtype=np.int32)
    dense_shape_b = np.array([5])
    sp_b = tf.sparse.SparseTensor(indices_b, values_b, dense_shape_b)

    input_dict = {"sp_a": sp_a, "sp_b": sp_b, "name": "minimum_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    indices_a = np.array([[0, 0, 0]])
    values_a = np.array([1], dtype=np.int32)
    dense_shape_a = np.array([2, 2, 2])
    sp_a = tf.sparse.SparseTensor(indices_a, values_a, dense_shape_a)

    indices_b = np.array([[0, 0, 0]])
    values_b = np.array([2], dtype=np.int32)
    dense_shape_b = np.array([2, 2, 2])
    sp_b = tf.sparse.SparseTensor(indices_b, values_b, dense_shape_b)

    input_dict = {"sp_a": sp_a, "sp_b": sp_b, "name": "minimum_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 4
    indices_a = np.array([[0, 0], [1, 1]])
    values_a = np.array([10, 20], dtype=np.int32)
    dense_shape_a = np.array([2, 2])
    sp_a = tf.sparse.SparseTensor(indices_a, values_a, dense_shape_a)

    indices_b = np.array([[0, 0], [1, 1]])
    values_b = np.array([5, 15], dtype=np.int32)
    dense_shape_b = np.array([2, 2])
    sp_b = tf.sparse.SparseTensor(indices_b, values_b, dense_shape_b)

    input_dict = {"sp_a": sp_a, "sp_b": sp_b, "name": "minimum_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    indices_a = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    values_a = np.array([1, 2, 3, 4], dtype=np.int32)
    dense_shape_a = np.array([2, 2])
    sp_a = tf.sparse.SparseTensor(indices_a, values_a, dense_shape_a)

    indices_b = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    values_b = np.array([5, 6, 7, 8], dtype=np.int32)
    dense_shape_b = np.array([2, 2])
    sp_b = tf.sparse.SparseTensor(indices_b, values_b, dense_shape_b)

    input_dict = {"sp_a": sp_a, "sp_b": sp_b, "name": "minimum_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    indices_a = np.array([[0, 0], [1, 0]])
    values_a = np.array([-5, -10], dtype=np.int32)
    dense_shape_a = np.array([2, 2])
    sp_a = tf.sparse.SparseTensor(indices_a, values_a, dense_shape_a)

    indices_b = np.array([[0, 0], [1, 1]])
    values_b = np.array([-1, -2], dtype=np.int32)
    dense_shape_b = np.array([2, 2])
    sp_b = tf.sparse.SparseTensor(indices_b, values_b, dense_shape_b)

    input_dict = {"sp_a": sp_a, "sp_b": sp_b, "name": "minimum_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    indices_a = np.array([[0, 0], [1, 0], [1, 1]], dtype=np.int64)
    values_a = np.array([1, 2, 3], dtype=np.int32)
    dense_shape_a = np.array([2, 2], dtype=np.int64)
    sp_a = tf.sparse.SparseTensor(indices_a, values_a, dense_shape_a)

    indices_b = np.array([[0, 0], [0, 1], [1, 1]], dtype=np.int64)
    values_b = np.array([4, 5, 6], dtype=np.int32)
    dense_shape_b = np.array([2, 2], dtype=np.int64)
    sp_b = tf.sparse.SparseTensor(indices_b, values_b, dense_shape_b)

    input_dict = {"sp_a": sp_a, "sp_b": sp_b, "name": "minimum_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    indices_a = np.array([[0], [1]])
    values_a = np.array([1, 2], dtype=np.int32)
    dense_shape_a = np.array([5])
    sp_a = tf.sparse.SparseTensor(indices_a, values_a, dense_shape_a)

    indices_b = np.array([[0], [1]])
    values_b = np.array([3, 1], dtype=np.int32)
    dense_shape_b = np.array([5])
    sp_b = tf.sparse.SparseTensor(indices_b, values_b, dense_shape_b)

    input_dict = {"sp_a": sp_a, "sp_b": sp_b, "name": "minimum_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 - Empty SparseTensors
    indices_a = np.empty((0, 2), dtype=np.int64)
    values_a = np.empty((0,), dtype=np.int32)
    dense_shape_a = np.array([2, 2], dtype=np.int64)
    sp_a = tf.sparse.SparseTensor(indices_a, values_a, dense_shape_a)

    indices_b = np.empty((0, 2), dtype=np.int64)
    values_b = np.empty((0,), dtype=np.int32)
    dense_shape_b = np.array([2, 2], dtype=np.int64)
    sp_b = tf.sparse.SparseTensor(indices_b, values_b, dense_shape_b)

    input_dict = {"sp_a": sp_a, "sp_b": sp_b, "name": "minimum_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    indices_a = np.array([[0, 0]])
    values_a = np.array([1], dtype=np.int32)
    dense_shape_a = np.array([1, 1])
    sp_a = tf.sparse.SparseTensor(indices_a, values_a, dense_shape_a)

    indices_b = np.array([[0, 0]])
    values_b = np.array([2], dtype=np.int32)
    dense_shape_b = np.array([1, 1])
    sp_b = tf.sparse.SparseTensor(indices_b, values_b, dense_shape_b)

    input_dict = {"sp_a": sp_a, "sp_b": sp_b, "name": "minimum_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.sparse.minimum"] = tf_sparse_minimum_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.sparse.minimum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.minimum'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.sparse.minimum', generated_inputs['tf.sparse.minimum'], lib="tf", suffix=0)
