
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_take_many_sparse_from_tensors_map_inputs():
    list_of_inputs = []

    # The function requires a SparseTensorsMap to be populated first.
    # Since we can't directly create a SparseTensorsMap within the input generation,
    # we'll create dummy inputs that satisfy the type and shape requirements.

    # It's likely that the runtime environment will have a pre-existing SparseTensorsMap
    # to which these handles will refer.  If not, the op will fail at runtime.
    # This generator cannot create a valid functional test case without external setup.

    # Input 1
    sparse_handles = np.array([0], dtype=np.int64)
    dtype = tf.float32
    container = ""
    shared_name = "test_sparse_map"  # A pre-existing SparseTensorsMap name
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
    sparse_handles = np.array([1], dtype=np.int64)
    dtype = tf.int32
    container = "my_container"  # A pre-existing SparseTensorsMap name
    shared_name = "another_sparse_map"
    name = "take_sparse_int"

    input_dict = {
        "sparse_handles": sparse_handles,
        "dtype": dtype,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    sparse_handles = np.array([2], dtype=np.int64)
    dtype = tf.float64
    container = ""
    shared_name = "sparse_map_float64"  # A pre-existing SparseTensorsMap name
    name = "take_sparse_float64"

    input_dict = {
        "sparse_handles": sparse_handles,
        "dtype": dtype,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    sparse_handles = np.array([3], dtype=np.int64)
    dtype = tf.int64
    container = "container_int64"  # A pre-existing SparseTensorsMap name
    shared_name = "sparse_map_int64"
    name = "take_sparse_int64"

    input_dict = {
        "sparse_handles": sparse_handles,
        "dtype": dtype,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 5
    sparse_handles = np.array([4], dtype=np.int64)
    dtype = tf.complex64
    container = ""
    shared_name = "sparse_map_complex64"  # A pre-existing SparseTensorsMap name
    name = "take_sparse_complex64"

    input_dict = {
        "sparse_handles": sparse_handles,
        "dtype": dtype,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6
    sparse_handles = np.array([5], dtype=np.int64)
    dtype = tf.complex128
    container = "container_complex128"  # A pre-existing SparseTensorsMap name
    shared_name = "sparse_map_complex128"
    name = "take_sparse_complex128"

    input_dict = {
        "sparse_handles": sparse_handles,
        "dtype": dtype,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7
    sparse_handles = np.array([6], dtype=np.int64)
    dtype = tf.bool
    container = ""
    shared_name = "sparse_map_bool"  # A pre-existing SparseTensorsMap name
    name = "take_sparse_bool"

    input_dict = {
        "sparse_handles": sparse_handles,
        "dtype": dtype,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 8
    sparse_handles = np.array([7], dtype=np.int64)
    dtype = tf.string
    container = "container_string"  # A pre-existing SparseTensorsMap name
    shared_name = "sparse_map_string"
    name = "take_sparse_string"

    input_dict = {
        "sparse_handles": sparse_handles,
        "dtype": dtype,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 9
    sparse_handles = np.array([8], dtype=np.int64)
    dtype = tf.float16
    container = ""
    shared_name = "sparse_map_float16"  # A pre-existing SparseTensorsMap name
    name = "take_sparse_float16"

    input_dict = {
        "sparse_handles": sparse_handles,
        "dtype": dtype,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    sparse_handles = np.array([9], dtype=np.int64)
    dtype = tf.int8
    container = "container_int8"  # A pre-existing SparseTensorsMap name
    shared_name = "sparse_map_int8"
    name = "take_sparse_int8"

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
