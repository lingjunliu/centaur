
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_quantized_mat_mul_inputs():
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
        "a": tf.constant(a, dtype=tf.qint8),
        "b": tf.constant(b, dtype=tf.qint8),
        "min_a": tf.constant(min_a, dtype=tf.float32),
        "max_a": tf.constant(max_a, dtype=tf.float32),
        "min_b": tf.constant(min_b, dtype=tf.float32),
        "max_b": tf.constant(max_b, dtype=tf.float32),
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
    min_a = np.float32(1.0)
    max_a = np.float32(3.0)
    min_b = np.float32(4.0)
    max_b = np.float32(6.0)
    Toutput = tf.qint16
    transpose_a = False
    transpose_b = False
    Tactivation = tf.qint8
    name = "matmul_2"
    input_dict = {
        "a": tf.constant(a, dtype=tf.quint8),
        "b": tf.constant(b, dtype=tf.quint8),
        "min_a": tf.constant(min_a, dtype=tf.float32),
        "max_a": tf.constant(max_a, dtype=tf.float32),
        "min_b": tf.constant(min_b, dtype=tf.float32),
        "max_b": tf.constant(max_b, dtype=tf.float32),
        "Toutput": Toutput,
        "transpose_a": transpose_a,
        "transpose_b": transpose_b,
        "Tactivation": Tactivation,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    a = np.array([[-1, 2], [3, -4]], dtype=np.int8)
    b = np.array([[5, -6], [-7, 8]], dtype=np.int8)
    min_a = np.float32(-5.0)
    max_a = np.float32(5.0)
    min_b = np.float32(-10.0)
    max_b = np.float32(10.0)
    Toutput = tf.qint32
    transpose_a = True
    transpose_b = False
    Tactivation = tf.quint8
    name = "matmul_3"
    input_dict = {
        "a": tf.constant(a, dtype=tf.qint8),
        "b": tf.constant(b, dtype=tf.qint8),
        "min_a": tf.constant(min_a, dtype=tf.float32),
        "max_a": tf.constant(max_a, dtype=tf.float32),
        "min_b": tf.constant(min_b, dtype=tf.float32),
        "max_b": tf.constant(max_b, dtype=tf.float32),
        "Toutput": Toutput,
        "transpose_a": transpose_a,
        "transpose_b": transpose_b,
        "Tactivation": Tactivation,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    a = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    b = np.array([[5, 6], [7, 8]], dtype=np.uint8)
    min_a = np.float32(0.0)
    max_a = np.float32(5.0)
    min_b = np.float32(0.0)
    max_b = np.float32(10.0)
    Toutput = tf.qint32
    transpose_a = False
    transpose_b = True
    Tactivation = tf.quint8
    name = "matmul_4"
    input_dict = {
        "a": tf.constant(a, dtype=tf.quint8),
        "b": tf.constant(b, dtype=tf.quint8),
        "min_a": tf.constant(min_a, dtype=tf.float32),
        "max_a": tf.constant(max_a, dtype=tf.float32),
        "min_b": tf.constant(min_b, dtype=tf.float32),
        "max_b": tf.constant(max_b, dtype=tf.float32),
        "Toutput": Toutput,
        "transpose_a": transpose_a,
        "transpose_b": transpose_b,
        "Tactivation": Tactivation,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 5
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int8)
    b = np.array([[7, 8], [9, 10], [11, 12]], dtype=np.int8)
    min_a = np.float32(-1.0)
    max_a = np.float32(6.0)
    min_b = np.float32(7.0)
    max_b = np.float32(12.0)
    Toutput = tf.qint32
    transpose_a = False
    transpose_b = False
    Tactivation = tf.quint8
    name = "matmul_5"
    input_dict = {
        "a": tf.constant(a, dtype=tf.qint8),
        "b": tf.constant(b, dtype=tf.qint8),
        "min_a": tf.constant(min_a, dtype=tf.float32),
        "max_a": tf.constant(max_a, dtype=tf.float32),
        "min_b": tf.constant(min_b, dtype=tf.float32),
        "max_b": tf.constant(max_b, dtype=tf.float32),
        "Toutput": Toutput,
        "transpose_a": transpose_a,
        "transpose_b": transpose_b,
        "Tactivation": Tactivation,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.QuantizedMatMul"] = tf_raw_ops_quantized_mat_mul_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.QuantizedMatMul' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QuantizedMatMul'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.QuantizedMatMul', generated_inputs['tf.raw_ops.QuantizedMatMul'], lib="tf", suffix=0)
