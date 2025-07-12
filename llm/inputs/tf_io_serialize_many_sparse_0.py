
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_serialize_many_sparse_inputs():
    list_of_inputs = []

    # Input 1: Basic case
    indices = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64)
    values = np.array([1, 2, 3, 4], dtype=np.float32)
    shape = np.array([2, 2], dtype=np.int64)
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=shape)
    out_type = tf.string
    name = "sparse_tensor_1"

    input_dict = {
        "sp_input": sp_input,
        "out_type": out_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different data type for values
    indices = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64)
    values = np.array([True, False, True, False], dtype=np.bool_)
    shape = np.array([2, 2], dtype=np.int64)
    sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=shape)
    out_type = tf.string
    name = "sparse_tensor_2"

    input_dict = {
        "sp_input": sp_input,
        "out_type": out_type,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.io.serialize_many_sparse"] = tf_io_serialize_many_sparse_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.io.serialize_many_sparse' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.serialize_many_sparse'.")

check_valid('tf.io.serialize_many_sparse', generated_inputs['tf.io.serialize_many_sparse'], lib="tf", suffix=0)
