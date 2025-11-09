
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_sparse_SparseTensor_2_inputs():
    list_of_inputs = []

    indices = [[0, 0], [1, 2], [2, 3]]
    values = np.array([1.5, -2.0, 3.25], dtype=np.float32)
    dense_shape = [3, 4]
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    indices = [[0], [3]]
    values = np.array([True, True], dtype=bool)
    dense_shape = [5]
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    indices = [[0, 0, 1], [1, 2, 3]]
    values = np.array([-10, 20], dtype=np.int32)
    dense_shape = [2, 3, 4]
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    indices = [[0, 1, 2], [1, 0, 0], [1, 2, 1], [0, 0, 0]]
    values = np.array([0.0, -3.14, 2.71828, 1.0], dtype=np.float64)
    dense_shape = [2, 3, 3]
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    indices = [[0, 0], [2, 2]]
    values = np.array([2**33, -(2**33)], dtype=np.int64)
    dense_shape = [3, 3]
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    indices = [[0]]
    values = np.array([np.nan], dtype=np.float32)
    dense_shape = [1]
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    indices = [[0, 0, 0], [1, 1, 1], [2, 2, 2], [0, 2, 1], [2, 0, 2]]
    values = np.array([np.inf, -np.inf, 0.0, 7.0, -1.0], dtype=np.float32)
    dense_shape = [3, 3, 3]
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    indices = [[0, 1], [1, 2]]
    values = np.array([255, 128], dtype=np.uint8)
    dense_shape = [2, 3]
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    indices = [[0, 0], [1, 1], [2, 2]]
    values = np.array([1.5, -2.5, 3.5], dtype=np.float16)
    dense_shape = [3, 3]
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    indices = [[0, 0, 0, 0], [1, 2, 0, 3], [1, 0, 0, 1], [0, 1, 0, 2]]
    values = np.array([-1000, 2000, 15, -7], dtype=np.int16)
    dense_shape = [2, 3, 1, 4]
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.sparse.SparseTensor_2"] = tf_sparse_SparseTensor_2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.sparse.SparseTensor_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.SparseTensor_2'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.sparse.SparseTensor', generated_inputs['tf.sparse.SparseTensor_2'], lib="tf", suffix=2)
