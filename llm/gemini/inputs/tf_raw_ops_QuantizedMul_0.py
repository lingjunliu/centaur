
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_QuantizedMul_inputs():
    list_of_inputs = []

    # Input 1
    x = np.array([1, 2, 3, 4], dtype=np.int8)
    y = np.array([5, 6, 7, 8], dtype=np.int8)
    min_x = np.array([0.0], dtype=np.float32)
    max_x = np.array([5.0], dtype=np.float32)
    min_y = np.array([0.0], dtype=np.float32)
    max_y = np.array([10.0], dtype=np.float32)
    Toutput = tf.qint32
    name = "quantized_mul_1"

    input_dict = {
        "x": tf.constant(x, dtype=tf.int8),
        "y": tf.constant(y, dtype=tf.int8),
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "Toutput": Toutput,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    x = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    y = np.array([[5, 6], [7, 8]], dtype=np.uint8)
    min_x = np.array([0.0], dtype=np.float32)
    max_x = np.array([255.0], dtype=np.float32)
    min_y = np.array([0.0], dtype=np.float32)
    max_y = np.array([255.0], dtype=np.float32)
    Toutput = tf.qint32
    name = "quantized_mul_2"

    input_dict = {
        "x": tf.constant(x, dtype=tf.uint8),
        "y": tf.constant(y, dtype=tf.uint8),
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "Toutput": Toutput,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    x = np.array([100, 200, 300], dtype=np.int32)
    y = np.array([500, 600, 700], dtype=np.int32)
    min_x = np.array([-100.0], dtype=np.float32)
    max_x = np.array([400.0], dtype=np.float32)
    min_y = np.array([-50.0], dtype=np.float32)
    max_y = np.array([800.0], dtype=np.float32)
    Toutput = tf.qint32
    name = "quantized_mul_3"

    input_dict = {
        "x": tf.constant(x, dtype=tf.int32),
        "y": tf.constant(y, dtype=tf.int32),
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "Toutput": Toutput,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    x = np.array([[-1, -2], [-3, -4]], dtype=np.int16)
    y = np.array([[5, 6], [7, 8]], dtype=np.int16)
    min_x = np.array([-5.0], dtype=np.float32)
    max_x = np.array([0.0], dtype=np.float32)
    min_y = np.array([0.0], dtype=np.float32)
    max_y = np.array([10.0], dtype=np.float32)
    Toutput = tf.qint32
    name = "quantized_mul_4"

    input_dict = {
        "x": tf.constant(x, dtype=tf.int16),
        "y": tf.constant(y, dtype=tf.int16),
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "Toutput": Toutput,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    x = np.array([1, 2, 3], dtype=np.uint16)
    y = np.array([5, 6, 7], dtype=np.uint16)
    min_x = np.array([0.0], dtype=np.float32)
    max_x = np.array([100.0], dtype=np.float32)
    min_y = np.array([0.0], dtype=np.float32)
    max_y = np.array([200.0], dtype=np.float32)
    Toutput = tf.qint32
    name = "quantized_mul_5"

    input_dict = {
        "x": tf.constant(x, dtype=tf.uint16),
        "y": tf.constant(y, dtype=tf.uint16),
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "Toutput": Toutput,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 6
    x = np.array([1, 2, 3, 4], dtype=np.int8)
    y = np.array([5, 6, 7, 8], dtype=np.int8)
    min_x = np.array([-5.0], dtype=np.float32)
    max_x = np.array([5.0], dtype=np.float32)
    min_y = np.array([-10.0], dtype=np.float32)
    max_y = np.array([10.0], dtype=np.float32)
    Toutput = tf.qint8
    name = "quantized_mul_6"

    input_dict = {
        "x": tf.constant(x, dtype=tf.int8),
        "y": tf.constant(y, dtype=tf.int8),
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "Toutput": Toutput,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    x = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    y = np.array([[5, 6], [7, 8]], dtype=np.uint8)
    min_x = np.array([0.0], dtype=np.float32)
    max_x = np.array([1.0], dtype=np.float32)
    min_y = np.array([0.0], dtype=np.float32)
    max_y = np.array([1.0], dtype=np.float32)
    Toutput = tf.quint8
    name = "quantized_mul_7"

    input_dict = {
        "x": tf.constant(x, dtype=tf.uint8),
        "y": tf.constant(y, dtype=tf.uint8),
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "Toutput": Toutput,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    x = np.array([100, 200, 300], dtype=np.int32)
    y = np.array([500, 600, 700], dtype=np.int32)
    min_x = np.array([-100.0], dtype=np.float32)
    max_x = np.array([400.0], dtype=np.float32)
    min_y = np.array([-50.0], dtype=np.float32)
    max_y = np.array([800.0], dtype=np.float32)
    Toutput = tf.qint16
    name = "quantized_mul_8"

    input_dict = {
        "x": tf.constant(x, dtype=tf.int32),
        "y": tf.constant(y, dtype=tf.int32),
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "Toutput": Toutput,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    x = np.array([[-1, -2], [-3, -4]], dtype=np.int16)
    y = np.array([[5, 6], [7, 8]], dtype=np.int16)
    min_x = np.array([-5.0], dtype=np.float32)
    max_x = np.array([0.0], dtype=np.float32)
    min_y = np.array([0.0], dtype=np.float32)
    max_y = np.array([10.0], dtype=np.float32)
    Toutput = tf.quint16
    name = "quantized_mul_9"

    input_dict = {
        "x": tf.constant(x, dtype=tf.int16),
        "y": tf.constant(y, dtype=tf.int16),
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "Toutput": Toutput,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    x = np.array([1, 2, 3], dtype=np.uint16)
    y = np.array([5, 6, 7], dtype=np.uint16)
    min_x = np.array([0.0], dtype=np.float32)
    max_x = np.array([100.0], dtype=np.float32)
    min_y = np.array([0.0], dtype=np.float32)
    max_y = np.array([200.0], dtype=np.float32)
    Toutput = tf.qint32
    name = "quantized_mul_10"

    input_dict = {
        "x": tf.constant(x, dtype=tf.uint16),
        "y": tf.constant(y, dtype=tf.uint16),
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "Toutput": Toutput,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.QuantizedMul"] = tf_raw_ops_QuantizedMul_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.QuantizedMul' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QuantizedMul'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.QuantizedMul', generated_inputs['tf.raw_ops.QuantizedMul'], lib="tf", suffix=0)
