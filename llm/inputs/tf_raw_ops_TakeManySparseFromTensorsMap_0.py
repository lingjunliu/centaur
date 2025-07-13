
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_take_many_sparse_from_tensors_map_inputs():
    list_of_inputs = []

    # Input 1
    sparse_handles = np.array([[1], [2]], dtype=np.int64)
    dtype = tf.float32
    container = ""
    shared_name = "test_sparse_tensors_map"
    name = "take_sparse"

    input_dict = {
        "sparse_handles": sparse_handles,
        "dtype": dtype,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    sparse_handles = np.array([[3]], dtype=np.int64)
    dtype = tf.int32
    container = "my_container"
    shared_name = "another_sparse_map"
    name = None

    input_dict = {
        "sparse_handles": sparse_handles,
        "dtype": dtype,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    sparse_handles = np.array([[10], [20], [30]], dtype=np.int64)
    dtype = tf.float64
    container = ""
    shared_name = "yet_another_map"
    name = "take_op"

    input_dict = {
        "sparse_handles": sparse_handles,
        "dtype": dtype,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    sparse_handles = np.array([[0]], dtype=np.int64)
    dtype = tf.int64
    container = "container_name"
    shared_name = "a_very_unique_name"
    name = "a_special_name"

    input_dict = {
        "sparse_handles": sparse_handles,
        "dtype": dtype,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 5
    sparse_handles = np.array([[123], [456], [789], [999]], dtype=np.int64)
    dtype = tf.uint8
    container = ""
    shared_name = "test_uint8"
    name = "take_uint8"

    input_dict = {
        "sparse_handles": sparse_handles,
        "dtype": dtype,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    sparse_handles = np.array([[ -1]], dtype=np.int64)
    dtype = tf.int16
    container = "neg_cont"
    shared_name = "neg_shared"
    name = "neg_name"

    input_dict = {
        "sparse_handles": sparse_handles,
        "dtype": dtype,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    sparse_handles = np.array([[-2], [-3]], dtype=np.int64)
    dtype = tf.uint16
    container = ""
    shared_name = "shared_uint16"
    name = "name_uint16"

    input_dict = {
        "sparse_handles": sparse_handles,
        "dtype": dtype,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    sparse_handles = np.array([[-4], [5], [-6]], dtype=np.int64)
    dtype = tf.int8
    container = "cont_int8"
    shared_name = "shared_int8"
    name = "name_int8"

    input_dict = {
        "sparse_handles": sparse_handles,
        "dtype": dtype,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    sparse_handles = np.array([[2**20]], dtype=np.int64)
    dtype = tf.complex64
    container = "complex_container"
    shared_name = "complex_shared_name"
    name = "complex_name"

    input_dict = {
        "sparse_handles": sparse_handles,
        "dtype": dtype,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    sparse_handles = np.array([[-(2**20)]], dtype=np.int64)
    dtype = tf.complex128
    container = ""
    shared_name = "complex128_shared"
    name = "complex128_name"

    input_dict = {
        "sparse_handles": sparse_handles,
        "dtype": dtype,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.TakeManySparseFromTensorsMap"] = tf_raw_ops_take_many_sparse_from_tensors_map_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.TakeManySparseFromTensorsMap' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.TakeManySparseFromTensorsMap'.")

check_valid('tf.raw_ops.TakeManySparseFromTensorsMap', generated_inputs['tf.raw_ops.TakeManySparseFromTensorsMap'], lib="tf", suffix=0)
