
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_sparse_maximum_inputs():
    list_of_inputs = []

    # Input 1
    indices_a = np.array([[0], [2]])
    values_a = np.array([1, 5], dtype=np.int32)
    dense_shape_a = np.array([5])
    sp_a = tf.sparse.SparseTensor(indices_a, values_a, dense_shape_a)

    indices_b = np.array([[1], [2]])
    values_b = np.array([2, 3], dtype=np.int32)
    dense_shape_b = np.array([5])
    sp_b = tf.sparse.SparseTensor(indices_b, values_b, dense_shape_b)

    input_dict = {"sp_a": sp_a, "sp_b": sp_b, "name": "sparse_max_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    indices_a = np.array([[0, 0], [0, 1]])
    values_a = np.array([1, 2], dtype=np.int32)
    dense_shape_a = np.array([2, 2])
    sp_a = tf.sparse.SparseTensor(indices_a, values_a, dense_shape_a)

    indices_b = np.array([[0, 1], [1, 0]])
    values_b = np.array([3, 4], dtype=np.int32)
    dense_shape_b = np.array([2, 2])
    sp_b = tf.sparse.SparseTensor(indices_b, values_b, dense_shape_b)

    input_dict = {"sp_a": sp_a, "sp_b": sp_b, "name": "sparse_max_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    indices_a = np.array([[0], [1], [2]])
    values_a = np.array([-1, -2, -3], dtype=np.int32)
    dense_shape_a = np.array([5])
    sp_a = tf.sparse.SparseTensor(indices_a, values_a, dense_shape_a)

    indices_b = np.array([[0], [2], [3]])
    values_b = np.array([-4, -5, -6], dtype=np.int32)
    dense_shape_b = np.array([5])
    sp_b = tf.sparse.SparseTensor(indices_b, values_b, dense_shape_b)

    input_dict = {"sp_a": sp_a, "sp_b": sp_b, "name": "sparse_max_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    indices_a = np.array([[0, 0], [1, 1]])
    values_a = np.array([1.5, 2.5], dtype=np.float32)
    dense_shape_a = np.array([3, 3])
    sp_a = tf.sparse.SparseTensor(indices_a, values_a, dense_shape_a)

    indices_b = np.array([[0, 1], [1, 1], [2, 2]])
    values_b = np.array([3.5, 0.5, 4.5], dtype=np.float32)
    dense_shape_b = np.array([3, 3])
    sp_b = tf.sparse.SparseTensor(indices_b, values_b, dense_shape_b)

    input_dict = {"sp_a": sp_a, "sp_b": sp_b, "name": "sparse_max_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    indices_a = np.array([[0, 0, 0], [0, 0, 1]])
    values_a = np.array([1, 2], dtype=np.int32)
    dense_shape_a = np.array([1, 1, 2])
    sp_a = tf.sparse.SparseTensor(indices_a, values_a, dense_shape_a)

    indices_b = np.array([[0, 0, 1]])
    values_b = np.array([3], dtype=np.int32)
    dense_shape_b = np.array([1, 1, 2])
    sp_b = tf.sparse.SparseTensor(indices_b, values_b, dense_shape_b)

    input_dict = {"sp_a": sp_a, "sp_b": sp_b, "name": "sparse_max_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    indices_a = np.array([[0]])
    values_a = np.array([1], dtype=np.int64)
    dense_shape_a = np.array([1])
    sp_a = tf.sparse.SparseTensor(indices_a, values_a, dense_shape_a)

    indices_b = np.array([[0]])
    values_b = np.array([2], dtype=np.int64)
    dense_shape_b = np.array([1])
    sp_b = tf.sparse.SparseTensor(indices_b, values_b, dense_shape_b)

    input_dict = {"sp_a": sp_a, "sp_b": sp_b, "name": "sparse_max_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    indices_a = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    values_a = np.array([1, 2, 3, 4], dtype=np.int32)
    dense_shape_a = np.array([2, 2])
    sp_a = tf.sparse.SparseTensor(indices_a, values_a, dense_shape_a)

    indices_b = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    values_b = np.array([5, 6, 7, 8], dtype=np.int32)
    dense_shape_b = np.array([2, 2])
    sp_b = tf.sparse.SparseTensor(indices_b, values_b, dense_shape_b)
    input_dict = {"sp_a": sp_a, "sp_b": sp_b, "name": "sparse_max_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    indices_a = np.array([[0, 0], [1, 1]])
    values_a = np.array([-1, -2], dtype=np.int32)
    dense_shape_a = np.array([2, 2])
    sp_a = tf.sparse.SparseTensor(indices_a, values_a, dense_shape_a)

    indices_b = np.array([[0, 1], [1, 0]])
    values_b = np.array([-3, -4], dtype=np.int32)
    dense_shape_b = np.array([2, 2])
    sp_b = tf.sparse.SparseTensor(indices_b, values_b, dense_shape_b)
    input_dict = {"sp_a": sp_a, "sp_b": sp_b, "name": "sparse_max_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    indices_a = np.array([[0], [1]])
    values_a = np.array([1, 2], dtype=np.float64)
    dense_shape_a = np.array([3])
    sp_a = tf.sparse.SparseTensor(indices_a, values_a, dense_shape_a)

    indices_b = np.array([[1], [2]])
    values_b = np.array([3, 4], dtype=np.float64)
    dense_shape_b = np.array([3])
    sp_b = tf.sparse.SparseTensor(indices_b, values_b, dense_shape_b)
    input_dict = {"sp_a": sp_a, "sp_b": sp_b, "name": "sparse_max_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    indices_a = np.array([[0, 0], [0, 1]])
    values_a = np.array([0, 0], dtype=np.int32)
    dense_shape_a = np.array([1, 2])
    sp_a = tf.sparse.SparseTensor(indices_a, values_a, dense_shape_a)

    indices_b = np.array([[0, 0], [0, 1]])
    values_b = np.array([0, 0], dtype=np.int32)
    dense_shape_b = np.array([1, 2])
    sp_b = tf.sparse.SparseTensor(indices_b, values_b, dense_shape_b)
    input_dict = {"sp_a": sp_a, "sp_b": sp_b, "name": "sparse_max_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
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


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.sparse.maximum', generated_inputs['tf.sparse.maximum'], lib="tf", suffix=0)
