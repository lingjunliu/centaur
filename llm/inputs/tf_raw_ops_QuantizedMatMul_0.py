
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_QuantizedMatMul_inputs():
    list_of_inputs = []

    # Input 1
    a = np.array([[1, 2], [3, 4]], dtype=np.int8)
    b = np.array([[5, 6], [7, 8]], dtype=np.int8)
    min_a = np.float32(0.0)
    max_a = np.float32(5.0)
    min_b = np.float32(0.0)
    max_b = np.float32(10.0)
    Toutput = tf.qint32
    transpose_a = False
    transpose_b = False
    Tactivation = tf.quint8
    name = "matmul_1"

    input_dict = {
        "a": a,
        "b": b,
        "min_a": min_a,
        "max_a": max_a,
        "min_b": min_b,
        "max_b": max_b,
        "Toutput": Toutput,
        "transpose_a": transpose_a,
        "transpose_b": transpose_b,
        "Tactivation": Tactivation,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    a = np.array([[1, 2, 3]], dtype=np.uint8)
    b = np.array([[4], [5], [6]], dtype=np.uint8)
    min_a = np.float32(-1.0)
    max_a = np.float32(2.0)
    min_b = np.float32(-3.0)
    max_b = np.float32(3.0)
    Toutput = tf.qint8
    transpose_a = True
    transpose_b = False
    Tactivation = tf.qint8
    name = "matmul_2"
    input_dict = {
        "a": a,
        "b": b,
        "min_a": min_a,
        "max_a": max_a,
        "min_b": min_b,
        "max_b": max_b,
        "Toutput": Toutput,
        "transpose_a": transpose_a,
        "transpose_b": transpose_b,
        "Tactivation": Tactivation,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    a = np.array([[1, 2], [3, 4]], dtype=np.int16)
    b = np.array([[5, 6], [7, 8]], dtype=np.int16)
    min_a = np.float32(-5.0)
    max_a = np.float32(0.0)
    min_b = np.float32(5.0)
    max_b = np.float32(15.0)
    Toutput = tf.quint16
    transpose_a = False
    transpose_b = True
    Tactivation = tf.quint16
    name = "matmul_3"
    input_dict = {
        "a": a,
        "b": b,
        "min_a": min_a,
        "max_a": max_a,
        "min_b": min_b,
        "max_b": max_b,
        "Toutput": Toutput,
        "transpose_a": transpose_a,
        "transpose_b": transpose_b,
        "Tactivation": Tactivation,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    b = np.array([[7, 8], [9, 10], [11, 12]], dtype=np.int32)
    min_a = np.float32(2.0)
    max_a = np.float32(8.0)
    min_b = np.float32(1.0)
    max_b = np.float32(7.0)
    Toutput = tf.qint32
    transpose_a = True
    transpose_b = True
    Tactivation = tf.qint32
    name = "matmul_4"
    input_dict = {
        "a": a,
        "b": b,
        "min_a": min_a,
        "max_a": max_a,
        "min_b": min_b,
        "max_b": max_b,
        "Toutput": Toutput,
        "transpose_a": transpose_a,
        "transpose_b": transpose_b,
        "Tactivation": Tactivation,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    a = np.array([[1, 2]], dtype=np.uint16)
    b = np.array([[3], [4]], dtype=np.uint16)
    min_a = np.float32(-1.0)
    max_a = np.float32(1.0)
    min_b = np.float32(-2.0)
    max_b = np.float32(2.0)
    Toutput = tf.quint8
    transpose_a = False
    transpose_b = False
    Tactivation = tf.quint8
    name = "matmul_5"
    input_dict = {
        "a": a,
        "b": b,
        "min_a": min_a,
        "max_a": max_a,
        "min_b": min_b,
        "max_b": max_b,
        "Toutput": Toutput,
        "transpose_a": transpose_a,
        "transpose_b": transpose_b,
        "Tactivation": Tactivation,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int8)
    b = np.array([[7, 8], [9, 10], [11, 12]], dtype=np.int8)
    min_a = np.float32(0.1)
    max_a = np.float32(0.9)
    min_b = np.float32(0.2)
    max_b = np.float32(0.8)
    Toutput = tf.qint32
    transpose_a = False
    transpose_b = False
    Tactivation = tf.quint8
    name = "matmul_6"
    input_dict = {
        "a": a,
        "b": b,
        "min_a": min_a,
        "max_a": max_a,
        "min_b": min_b,
        "max_b": max_b,
        "Toutput": Toutput,
        "transpose_a": transpose_a,
        "transpose_b": transpose_b,
        "Tactivation": Tactivation,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    a = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.uint8)
    b = np.array([[7, 8, 9], [10, 11, 12]], dtype=np.uint8)
    min_a = np.float32(-0.5)
    max_a = np.float32(0.5)
    min_b = np.float32(-1.0)
    max_b = np.float32(1.0)
    Toutput = tf.qint8
    transpose_a = False
    transpose_b = False
    Tactivation = tf.qint8
    name = "matmul_7"
    input_dict = {
        "a": a,
        "b": b,
        "min_a": min_a,
        "max_a": max_a,
        "min_b": min_b,
        "max_b": max_b,
        "Toutput": Toutput,
        "transpose_a": transpose_a,
        "transpose_b": transpose_b,
        "Tactivation": Tactivation,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    a = np.array([[1, 2, 3]], dtype=np.int16)
    b = np.array([[4], [5], [6]], dtype=np.int16)
    min_a = np.float32(10.0)
    max_a = np.float32(20.0)
    min_b = np.float32(30.0)
    max_b = np.float32(40.0)
    Toutput = tf.quint16
    transpose_a = False
    transpose_b = False
    Tactivation = tf.quint16
    name = "matmul_8"
    input_dict = {
        "a": a,
        "b": b,
        "min_a": min_a,
        "max_a": max_a,
        "min_b": min_b,
        "max_b": max_b,
        "Toutput": Toutput,
        "transpose_a": transpose_a,
        "transpose_b": transpose_b,
        "Tactivation": Tactivation,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    a = np.array([[1], [2], [3]], dtype=np.int32)
    b = np.array([[4, 5, 6]], dtype=np.int32)
    min_a = np.float32(-10.0)
    max_a = np.float32(-5.0)
    min_b = np.float32(-20.0)
    max_b = np.float32(-15.0)
    Toutput = tf.qint32
    transpose_a = False
    transpose_b = False
    Tactivation = tf.qint32
    name = "matmul_9"
    input_dict = {
        "a": a,
        "b": b,
        "min_a": min_a,
        "max_a": max_a,
        "min_b": min_b,
        "max_b": max_b,
        "Toutput": Toutput,
        "transpose_a": transpose_a,
        "transpose_b": transpose_b,
        "Tactivation": Tactivation,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint16)
    b = np.array([[7, 8], [9, 10], [11, 12]], dtype=np.uint16)
    min_a = np.float32(1.5)
    max_a = np.float32(2.5)
    min_b = np.float32(3.5)
    max_b = np.float32(4.5)
    Toutput = tf.quint8
    transpose_a = False
    transpose_b = False
    Tactivation = tf.quint8
    name = "matmul_10"
    input_dict = {
        "a": a,
        "b": b,
        "min_a": min_a,
        "max_a": max_a,
        "min_b": min_b,
        "max_b": max_b,
        "Toutput": Toutput,
        "transpose_a": transpose_a,
        "transpose_b": transpose_b,
        "Tactivation": Tactivation,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.QuantizedMatMul"] = tf_raw_ops_QuantizedMatMul_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.QuantizedMatMul' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QuantizedMatMul'.")

check_valid('tf.raw_ops.QuantizedMatMul', generated_inputs['tf.raw_ops.QuantizedMatMul'], lib="tf", suffix=0)
