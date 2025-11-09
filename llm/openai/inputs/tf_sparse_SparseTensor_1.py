
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def tf_sparse_SparseTensor_inputs():
    list_of_inputs = []

    indices = np.array([[0, 0], [1, 2]], dtype=np.int64)
    values = np.array([1, 2], dtype=np.int32)
    dense_shape = np.array([3, 4], dtype=np.int64)
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    indices = np.array([[3], [1], [5]], dtype=np.int64)
    values = np.array([10.0, -2.0, 5.0], dtype=np.float32)
    dense_shape = np.array([7], dtype=np.int64)
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    indices = np.array([[0, 0, 0], [1, 2, 3], [1, 0, 2]], dtype=np.int64)
    values = np.array([0.5, -1.25, 3.0], dtype=np.float64)
    dense_shape = np.array([2, 3, 4], dtype=np.int64)
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    indices = np.array([[0, 1]], dtype=np.int64)
    values = np.array([True], dtype=bool)
    dense_shape = np.array([2, 2], dtype=np.int64)
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    indices = np.array([[0], [2]], dtype=np.int64)
    values = np.array([1 + 2j, -3 + 0.5j], dtype=np.complex64)
    dense_shape = np.array([4], dtype=np.int64)
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    indices = np.empty((0, 2), dtype=np.int64)
    values = np.empty((0,), dtype=np.float32)
    dense_shape = np.array([0, 5], dtype=np.int64)
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    indices = np.array([[1, 1], [1, 1], [0, 2]], dtype=np.int64)
    values = np.array([5, -3, 9], dtype=np.int64)
    dense_shape = np.array([3, 3], dtype=np.int64)
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    indices = np.array([[0, 0, 0, 1], [1, 1, 1, 1]], dtype=np.int64)
    values = np.array([1.5, -2.5], dtype=np.float16)
    dense_shape = np.array([2, 2, 2, 2], dtype=np.int64)
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    indices = np.array([[999999]], dtype=np.int64)
    values = np.array([7], dtype=np.int8)
    dense_shape = np.array([1000000], dtype=np.int64)
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    indices = np.array([[0, 0], [1, 2]], dtype=np.int64)
    values = np.array([255, 128], dtype=np.uint8)
    dense_shape = np.array([2, 3], dtype=np.int64)
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    indices = np.array([[0, 0, 0, 0, 0]], dtype=np.int64)
    values = np.array([42], dtype=np.int16)
    dense_shape = np.array([1, 1, 1, 1, 1], dtype=np.int64)
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.sparse.SparseTensor_1"] = tf_sparse_SparseTensor_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.sparse.SparseTensor_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.SparseTensor_1'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.sparse.SparseTensor', generated_inputs['tf.sparse.SparseTensor_1'], lib="tf", suffix=1)
