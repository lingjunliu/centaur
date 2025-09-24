
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_make_tensor_proto_inputs():
    list_of_inputs = []

    # Input 1: Basic numpy array with default dtype and shape
    values = np.array([[1, 2], [3, 4]], dtype=np.int32)
    dtype = tf.dtypes.int32
    shape = [2, 2]
    verify_shape = True
    allow_broadcast = False
    input_dict = {"values": values, "dtype": dtype, "shape": shape, "verify_shape": verify_shape, "allow_broadcast": allow_broadcast}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2:  Float numpy array, different shape
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    dtype = tf.dtypes.float32
    shape = [5]
    verify_shape = True
    allow_broadcast = False
    input_dict = {"values": values, "dtype": dtype, "shape": shape, "verify_shape": verify_shape, "allow_broadcast": allow_broadcast}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Boolean numpy array
    values = np.array([[True, False], [False, True]], dtype=np.bool_)
    dtype = tf.dtypes.bool
    shape = [2, 2]
    verify_shape = True
    allow_broadcast = False
    input_dict = {"values": values, "dtype": dtype, "shape": shape, "verify_shape": verify_shape, "allow_broadcast": allow_broadcast}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Complex64 numpy array
    values = np.array([1+1j, 2+2j], dtype=np.complex64)
    dtype = tf.dtypes.complex64
    shape = [2]
    verify_shape = True
    allow_broadcast = False
    input_dict = {"values": values, "dtype": dtype, "shape": shape, "verify_shape": verify_shape, "allow_broadcast": allow_broadcast}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Int64 numpy array, 3D tensor
    values = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64)
    dtype = tf.dtypes.int64
    shape = [2, 2, 2]
    verify_shape = True
    allow_broadcast = False
    input_dict = {"values": values, "dtype": dtype, "shape": shape, "verify_shape": verify_shape, "allow_broadcast": allow_broadcast}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6:  Float64 numpy array with allow_broadcast True and scalar value
    values = np.array(5.0, dtype=np.float64)
    dtype = tf.dtypes.float64
    shape = [2, 2]
    verify_shape = False
    allow_broadcast = True
    input_dict = {"values": values, "dtype": dtype, "shape": shape, "verify_shape": verify_shape, "allow_broadcast": allow_broadcast}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Int32, 1D, allow_broadcast True
    values = np.array([1], dtype=np.int32)
    dtype = tf.dtypes.int32
    shape = [5]
    verify_shape = False
    allow_broadcast = True
    input_dict = {"values": values, "dtype": dtype, "shape": shape, "verify_shape": verify_shape, "allow_broadcast": allow_broadcast}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: uint8, 2D, with shape
    values = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    dtype = tf.dtypes.uint8
    shape = [2, 2]
    verify_shape = True
    allow_broadcast = False
    input_dict = {"values": values, "dtype": dtype, "shape": shape, "verify_shape": verify_shape, "allow_broadcast": allow_broadcast}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: int16, 3D
    values = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int16)
    dtype = tf.dtypes.int16
    shape = [2, 2, 2]
    verify_shape = True
    allow_broadcast = False
    input_dict = {"values": values, "dtype": dtype, "shape": shape, "verify_shape": verify_shape, "allow_broadcast": allow_broadcast}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: negative values, float32
    values = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32)
    dtype = tf.dtypes.float32
    shape = [2, 2]
    verify_shape = True
    allow_broadcast = False
    input_dict = {"values": values, "dtype": dtype, "shape": shape, "verify_shape": verify_shape, "allow_broadcast": allow_broadcast}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: Int8 numpy array
    values = np.array([[1, 2], [3, 4]], dtype=np.int8)
    dtype = tf.dtypes.int8
    shape = [2, 2]
    verify_shape = True
    allow_broadcast = False
    input_dict = {"values": values, "dtype": dtype, "shape": shape, "verify_shape": verify_shape, "allow_broadcast": allow_broadcast}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.make_tensor_proto"] = tf_make_tensor_proto_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.make_tensor_proto' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.make_tensor_proto'.")

check_valid('tf.make_tensor_proto', generated_inputs['tf.make_tensor_proto'], lib="tf", suffix=0)
