
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_cross_inputs():
    list_of_inputs = []

    # Input 1: Simple 3-element vectors, float32
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    input_dict = {"a": tf.convert_to_tensor(a, dtype=tf.float32), "b": tf.convert_to_tensor(b, dtype=tf.float32), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Simple 3-element vectors, int32
    a = np.array([1, 2, 3], dtype=np.int32)
    b = np.array([4, 5, 6], dtype=np.int32)
    input_dict = {"a": tf.convert_to_tensor(a, dtype=tf.int32), "b": tf.convert_to_tensor(b, dtype=tf.int32), "name": "cross_product_int"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Matrix where the innermost dimension is 3, float64
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    b = np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype=np.float64)
    input_dict = {"a": tf.convert_to_tensor(a, dtype=tf.float64), "b": tf.convert_to_tensor(b, dtype=tf.float64), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D tensor, int64
    a = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.int64)
    b = np.array([[[13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24]]], dtype=np.int64)
    input_dict = {"a": tf.convert_to_tensor(a, dtype=tf.int64), "b": tf.convert_to_tensor(b, dtype=tf.int64), "name": "3d_cross"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Simple 3-element vectors, float16
    a = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    b = np.array([4.0, 5.0, 6.0], dtype=np.float16)
    input_dict = {"a": tf.convert_to_tensor(a, dtype=tf.float16), "b": tf.convert_to_tensor(b, dtype=tf.float16), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Matrix where the innermost dimension is 3, half (float16)
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float16)
    b = np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype=np.float16)
    input_dict = {"a": tf.convert_to_tensor(a, dtype=tf.float16), "b": tf.convert_to_tensor(b, dtype=tf.float16), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: uint8
    a = np.array([1, 2, 3], dtype=np.uint8)
    b = np.array([4, 5, 6], dtype=np.uint8)
    input_dict = {"a": tf.convert_to_tensor(a, dtype=tf.uint8), "b": tf.convert_to_tensor(b, dtype=tf.uint8), "name": "cross_uint8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: uint16
    a = np.array([1, 2, 3], dtype=np.uint16)
    b = np.array([4, 5, 6], dtype=np.uint16)
    input_dict = {"a": tf.convert_to_tensor(a, dtype=tf.uint16), "b": tf.convert_to_tensor(b, dtype=tf.uint16), "name": "cross_uint16"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: uint32
    a = np.array([1, 2, 3], dtype=np.uint32)
    b = np.array([4, 5, 6], dtype=np.uint32)
    input_dict = {"a": tf.convert_to_tensor(a, dtype=tf.uint32), "b": tf.convert_to_tensor(b, dtype=tf.uint32), "name": "cross_uint32"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: uint64
    a = np.array([1, 2, 3], dtype=np.uint64)
    b = np.array([4, 5, 6], dtype=np.uint64)
    input_dict = {"a": tf.convert_to_tensor(a, dtype=tf.uint64), "b": tf.convert_to_tensor(b, dtype=tf.uint64), "name": "cross_uint64"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Cross"] = tf_raw_ops_cross_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Cross' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Cross'.")

check_valid('tf.raw_ops.Cross', generated_inputs['tf.raw_ops.Cross'], lib="tf", suffix=0)
