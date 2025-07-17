
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_QuantizedBiasAdd_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int8)
    bias_tensor = np.array([1, 2, 3], dtype=np.int8)
    min_input_tensor = np.float32(0.0)
    max_input_tensor = np.float32(6.0)
    min_bias_tensor = np.float32(1.0)
    max_bias_tensor = np.float32(3.0)
    out_type = tf.qint8
    input_dict = {"input": tf.constant(input_tensor, dtype=tf.qint8), "bias": tf.constant(bias_tensor, dtype=tf.qint8), "min_input": tf.constant(min_input_tensor, dtype=tf.float32), "max_input": tf.constant(max_input_tensor, dtype=tf.float32), "min_bias": tf.constant(min_bias_tensor, dtype=tf.float32), "max_bias": tf.constant(max_bias_tensor, dtype=tf.float32), "out_type": out_type, "name": "test1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    bias_tensor = np.array([1, 2, 3], dtype=np.uint8)
    min_input_tensor = np.float32(0.0)
    max_input_tensor = np.float32(255.0)
    min_bias_tensor = np.float32(0.0)
    max_bias_tensor = np.float32(255.0)
    out_type = tf.quint8
    input_dict = {"input": tf.constant(input_tensor, dtype=tf.quint8), "bias": tf.constant(bias_tensor, dtype=tf.quint8), "min_input": tf.constant(min_input_tensor, dtype=tf.float32), "max_input": tf.constant(max_input_tensor, dtype=tf.float32), "min_bias": tf.constant(min_bias_tensor, dtype=tf.float32), "max_bias": tf.constant(max_bias_tensor, dtype=tf.float32), "out_type": out_type, "name": "test2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    bias_tensor = np.array([1, 2, 3], dtype=np.int32)
    min_input_tensor = np.float32(-100.0)
    max_input_tensor = np.float32(100.0)
    min_bias_tensor = np.float32(-50.0)
    max_bias_tensor = np.float32(50.0)
    out_type = tf.qint32
    input_dict = {"input": tf.constant(input_tensor, dtype=tf.qint32), "bias": tf.constant(bias_tensor, dtype=tf.qint32), "min_input": tf.constant(min_input_tensor, dtype=tf.float32), "max_input": tf.constant(max_input_tensor, dtype=tf.float32), "min_bias": tf.constant(min_bias_tensor, dtype=tf.float32), "max_bias": tf.constant(max_bias_tensor, dtype=tf.float32), "out_type": out_type, "name": "test3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int16)
    bias_tensor = np.array([1, 2, 3], dtype=np.int16)
    min_input_tensor = np.float32(-10.0)
    max_input_tensor = np.float32(10.0)
    min_bias_tensor = np.float32(-5.0)
    max_bias_tensor = np.float32(5.0)
    out_type = tf.qint16
    input_dict = {"input": tf.constant(input_tensor, dtype=tf.qint16), "bias": tf.constant(bias_tensor, dtype=tf.qint16), "min_input": tf.constant(min_input_tensor, dtype=tf.float32), "max_input": tf.constant(max_input_tensor, dtype=tf.float32), "min_bias": tf.constant(min_bias_tensor, dtype=tf.float32), "max_bias": tf.constant(max_bias_tensor, dtype=tf.float32), "out_type": out_type, "name": "test4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint16)
    bias_tensor = np.array([1, 2, 3], dtype=np.uint16)
    min_input_tensor = np.float32(0.0)
    max_input_tensor = np.float32(65535.0)
    min_bias_tensor = np.float32(0.0)
    max_bias_tensor = np.float32(100.0)
    out_type = tf.quint16
    input_dict = {"input": tf.constant(input_tensor, dtype=tf.quint16), "bias": tf.constant(bias_tensor, dtype=tf.quint16), "min_input": tf.constant(min_input_tensor, dtype=tf.float32), "max_input": tf.constant(max_input_tensor, dtype=tf.float32), "min_bias": tf.constant(min_bias_tensor, dtype=tf.float32), "max_bias": tf.constant(max_bias_tensor, dtype=tf.float32), "out_type": out_type, "name": "test5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, different shape
    input_tensor = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.int8)
    bias_tensor = np.array([1, 2, 3], dtype=np.int8)
    min_input_tensor = np.float32(-128.0)
    max_input_tensor = np.float32(127.0)
    min_bias_tensor = np.float32(-10.0)
    max_bias_tensor = np.float32(10.0)
    out_type = tf.qint8
    input_dict = {"input": tf.constant(input_tensor, dtype=tf.qint8), "bias": tf.constant(bias_tensor, dtype=tf.qint8), "min_input": tf.constant(min_input_tensor, dtype=tf.float32), "max_input": tf.constant(max_input_tensor, dtype=tf.float32), "min_bias": tf.constant(min_bias_tensor, dtype=tf.float32), "max_bias": tf.constant(max_bias_tensor, dtype=tf.float32), "out_type": out_type, "name": "test6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, negative bias
    input_tensor = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int8)
    bias_tensor = np.array([-1, -2, -3], dtype=np.int8)
    min_input_tensor = np.float32(-5.0)
    max_input_tensor = np.float32(10.0)
    min_bias_tensor = np.float32(-3.0)
    max_bias_tensor = np.float32(-1.0)
    out_type = tf.qint8
    input_dict = {"input": tf.constant(input_tensor, dtype=tf.qint8), "bias": tf.constant(bias_tensor, dtype=tf.qint8), "min_input": tf.constant(min_input_tensor, dtype=tf.float32), "max_input": tf.constant(max_input_tensor, dtype=tf.float32), "min_bias": tf.constant(min_bias_tensor, dtype=tf.float32), "max_bias": tf.constant(max_bias_tensor, dtype=tf.float32), "out_type": out_type, "name": "test7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8, larger input values
    input_tensor = np.array([[100, 200, 300], [400, 500, 600]], dtype=np.int32)
    bias_tensor = np.array([10, 20, 30], dtype=np.int32)
    min_input_tensor = np.float32(100.0)
    max_input_tensor = np.float32(600.0)
    min_bias_tensor = np.float32(10.0)
    max_bias_tensor = np.float32(30.0)
    out_type = tf.qint32
    input_dict = {"input": tf.constant(input_tensor, dtype=tf.qint32), "bias": tf.constant(bias_tensor, dtype=tf.qint32), "min_input": tf.constant(min_input_tensor, dtype=tf.float32), "max_input": tf.constant(max_input_tensor, dtype=tf.float32), "min_bias": tf.constant(min_bias_tensor, dtype=tf.float32), "max_bias": tf.constant(max_bias_tensor, dtype=tf.float32), "out_type": out_type, "name": "test8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, zero min/max values
    input_tensor = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int8)
    bias_tensor = np.array([1, 2, 3], dtype=np.int8)
    min_input_tensor = np.float32(0.0)
    max_input_tensor = np.float32(0.0)
    min_bias_tensor = np.float32(0.0)
    max_bias_tensor = np.float32(0.0)
    out_type = tf.qint8
    input_dict = {"input": tf.constant(input_tensor, dtype=tf.qint8), "bias": tf.constant(bias_tensor, dtype=tf.qint8), "min_input": tf.constant(min_input_tensor, dtype=tf.float32), "max_input": tf.constant(max_input_tensor, dtype=tf.float32), "min_bias": tf.constant(min_bias_tensor, dtype=tf.float32), "max_bias": tf.constant(max_bias_tensor, dtype=tf.float32), "out_type": out_type, "name": "test9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 10, 3D input
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int8)
    bias_tensor = np.array([1, 2], dtype=np.int8)
    min_input_tensor = np.float32(-10.0)
    max_input_tensor = np.float32(10.0)
    min_bias_tensor = np.float32(-2.0)
    max_bias_tensor = np.float32(2.0)
    out_type = tf.qint8
    input_dict = {"input": tf.constant(input_tensor, dtype=tf.qint8), "bias": tf.constant(bias_tensor, dtype=tf.qint8), "min_input": tf.constant(min_input_tensor, dtype=tf.float32), "max_input": tf.constant(max_input_tensor, dtype=tf.float32), "min_bias": tf.constant(min_bias_tensor, dtype=tf.float32), "max_bias": tf.constant(max_bias_tensor, dtype=tf.float32), "out_type": out_type, "name": "test10"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11, quint16
    input_tensor = np.array([[1000, 2000, 3000], [4000, 5000, 6000]], dtype=np.uint16)
    bias_tensor = np.array([100, 200, 300], dtype=np.uint16)
    min_input_tensor = np.float32(0.0)
    max_input_tensor = np.float32(6000.0)
    min_bias_tensor = np.float32(0.0)
    max_bias_tensor = np.float32(300.0)
    out_type = tf.quint16
    input_dict = {"input": tf.constant(input_tensor, dtype=tf.quint16), "bias": tf.constant(bias_tensor, dtype=tf.quint16), "min_input": tf.constant(min_input_tensor, dtype=tf.float32), "max_input": tf.constant(max_input_tensor, dtype=tf.float32), "min_bias": tf.constant(min_bias_tensor, dtype=tf.float32), "max_bias": tf.constant(max_bias_tensor, dtype=tf.float32), "out_type": out_type, "name": "test11"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.QuantizedBiasAdd"] = tf_raw_ops_QuantizedBiasAdd_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.QuantizedBiasAdd' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QuantizedBiasAdd'.")

check_valid('tf.raw_ops.QuantizedBiasAdd', generated_inputs['tf.raw_ops.QuantizedBiasAdd'], lib="tf", suffix=0)
