
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_QuantizedAdd_inputs():
    list_of_inputs = []

    # Input 1
    x = np.array([1, 2, 3], dtype=np.int8)
    y = np.array([4, 5, 6], dtype=np.int8)
    min_x = np.float32(0.0)
    max_x = np.float32(5.0)
    min_y = np.float32(0.0)
    max_y = np.float32(10.0)
    Toutput = tf.qint32
    name = "quantized_add_1"
    x = tf.constant(x, dtype=tf.qint8)
    y = tf.constant(y, dtype=tf.qint8)
    input_dict = {
        "x": x,
        "y": y,
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
    min_x = np.float32(0.0)
    max_x = np.float32(10.0)
    min_y = np.float32(0.0)
    max_y = np.float32(15.0)
    Toutput = tf.qint32
    name = "quantized_add_2"
    x = tf.constant(x, dtype=tf.quint8)
    y = tf.constant(y, dtype=tf.quint8)

    input_dict = {
        "x": x,
        "y": y,
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "Toutput": Toutput,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    x = np.array([1, 2, 3, 4], dtype=np.int32)
    y = np.array([5, 6, 7, 8], dtype=np.int32)
    min_x = np.float32(-10.0)
    max_x = np.float32(10.0)
    min_y = np.float32(-15.0)
    max_y = np.float32(15.0)
    Toutput = tf.qint32
    name = "quantized_add_3"
    x = tf.constant(x, dtype=tf.qint32)
    y = tf.constant(y, dtype=tf.qint32)

    input_dict = {
        "x": x,
        "y": y,
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
    min_x = np.float32(-20.0)
    max_x = np.float32(-1.0)
    min_y = np.float32(1.0)
    max_y = np.float32(20.0)
    Toutput = tf.qint32
    name = "quantized_add_4"
    x = tf.constant(x, dtype=tf.qint16)
    y = tf.constant(y, dtype=tf.qint16)

    input_dict = {
        "x": x,
        "y": y,
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
    y = np.array([4, 5, 6], dtype=np.uint16)
    min_x = np.float32(0.0)
    max_x = np.float32(5.0)
    min_y = np.float32(0.0)
    max_y = np.float32(10.0)
    Toutput = tf.qint32
    name = "quantized_add_5"
    x = tf.constant(x, dtype=tf.quint16)
    y = tf.constant(y, dtype=tf.quint16)

    input_dict = {
        "x": x,
        "y": y,
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "Toutput": Toutput,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    x = np.array([1, 2, 3], dtype=np.int8)
    y = np.array([4, 5, 6], dtype=np.int8)
    min_x = np.float32(-5.0)
    max_x = np.float32(5.0)
    min_y = np.float32(-10.0)
    max_y = np.float32(10.0)
    Toutput = tf.qint32
    name = "quantized_add_6"
    x = tf.constant(x, dtype=tf.qint8)
    y = tf.constant(y, dtype=tf.qint8)

    input_dict = {
        "x": x,
        "y": y,
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
    min_x = np.float32(5.0)
    max_x = np.float32(10.0)
    min_y = np.float32(10.0)
    max_y = np.float32(15.0)
    Toutput = tf.qint32
    name = "quantized_add_7"
    x = tf.constant(x, dtype=tf.quint8)
    y = tf.constant(y, dtype=tf.quint8)

    input_dict = {
        "x": x,
        "y": y,
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "Toutput": Toutput,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    x = np.array([1, 2, 3, 4], dtype=np.int16)
    y = np.array([5, 6, 7, 8], dtype=np.int16)
    min_x = np.float32(-10.0)
    max_x = np.float32(10.0)
    min_y = np.float32(-15.0)
    max_y = np.float32(15.0)
    Toutput = tf.qint32
    name = "quantized_add_8"
    x = tf.constant(x, dtype=tf.qint16)
    y = tf.constant(y, dtype=tf.qint16)

    input_dict = {
        "x": x,
        "y": y,
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
    min_x = np.float32(-20.0)
    max_x = np.float32(-1.0)
    min_y = np.float32(1.0)
    max_y = np.float32(20.0)
    Toutput = tf.qint32
    name = "quantized_add_9"
    x = tf.constant(x, dtype=tf.qint16)
    y = tf.constant(y, dtype=tf.qint16)

    input_dict = {
        "x": x,
        "y": y,
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
    y = np.array([4, 5, 6], dtype=np.uint16)
    min_x = np.float32(0.0)
    max_x = np.float32(5.0)
    min_y = np.float32(0.0)
    max_y = np.float32(10.0)
    Toutput = tf.qint32
    name = "quantized_add_10"
    x = tf.constant(x, dtype=tf.quint16)
    y = tf.constant(y, dtype=tf.quint16)

    input_dict = {
        "x": x,
        "y": y,
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
generated_inputs["tf.raw_ops.QuantizedAdd"] = tf_raw_ops_QuantizedAdd_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.QuantizedAdd' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QuantizedAdd'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.QuantizedAdd', generated_inputs['tf.raw_ops.QuantizedAdd'], lib="tf", suffix=0)
