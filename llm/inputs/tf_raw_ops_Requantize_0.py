
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_requantize_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([[1, 2], [3, 4]], dtype=np.int8)
    input_min = np.array(-1.0, dtype=np.float32)
    input_max = np.array(1.0, dtype=np.float32)
    requested_output_min = np.array(-0.5, dtype=np.float32)
    requested_output_max = np.array(0.5, dtype=np.float32)
    out_type = tf.qint8

    input_dict = {
        "input": input_tensor,
        "input_min": input_min,
        "input_max": input_max,
        "requested_output_min": requested_output_min,
        "requested_output_max": requested_output_max,
        "out_type": out_type,
        "name": "RequantizeExample1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[-1, 0, 1]], dtype=np.int32)
    input_min = np.array(-5.0, dtype=np.float32)
    input_max = np.array(5.0, dtype=np.float32)
    requested_output_min = np.array(-2.0, dtype=np.float32)
    requested_output_max = np.array(2.0, dtype=np.float32)
    out_type = tf.qint8

    input_dict = {
        "input": input_tensor,
        "input_min": input_min,
        "input_max": input_max,
        "requested_output_min": requested_output_min,
        "requested_output_max": requested_output_max,
        "out_type": out_type,
        "name": "RequantizeExample2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([255], dtype=np.uint8)
    input_min = np.array(0.0, dtype=np.float32)
    input_max = np.array(255.0, dtype=np.float32)
    requested_output_min = np.array(0.0, dtype=np.float32)
    requested_output_max = np.array(100.0, dtype=np.float32)
    out_type = tf.qint8

    input_dict = {
        "input": input_tensor,
        "input_min": input_min,
        "input_max": input_max,
        "requested_output_min": requested_output_min,
        "requested_output_max": requested_output_max,
        "out_type": out_type,
        "name": "RequantizeExample3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int16)
    input_min = np.array(-10.0, dtype=np.float32)
    input_max = np.array(10.0, dtype=np.float32)
    requested_output_min = np.array(-5.0, dtype=np.float32)
    requested_output_max = np.array(5.0, dtype=np.float32)
    out_type = tf.qint8

    input_dict = {
        "input": input_tensor,
        "input_min": input_min,
        "input_max": input_max,
        "requested_output_min": requested_output_min,
        "requested_output_max": requested_output_max,
        "out_type": out_type,
        "name": "RequantizeExample4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([65535], dtype=np.uint16)
    input_min = np.array(0.0, dtype=np.float32)
    input_max = np.array(65535.0, dtype=np.float32)
    requested_output_min = np.array(0.0, dtype=np.float32)
    requested_output_max = np.array(255.0, dtype=np.float32)
    out_type = tf.quint8

    input_dict = {
        "input": input_tensor,
        "input_min": input_min,
        "input_max": input_max,
        "requested_output_min": requested_output_min,
        "requested_output_max": requested_output_max,
        "out_type": out_type,
        "name": "RequantizeExample5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array([-128], dtype=np.int8)
    input_min = np.array(-128.0, dtype=np.float32)
    input_max = np.array(127.0, dtype=np.float32)
    requested_output_min = np.array(-64.0, dtype=np.float32)
    requested_output_max = np.array(63.0, dtype=np.float32)
    out_type = tf.qint8

    input_dict = {
        "input": input_tensor,
        "input_min": input_min,
        "input_max": input_max,
        "requested_output_min": requested_output_min,
        "requested_output_max": requested_output_max,
        "out_type": out_type,
        "name": "RequantizeExample6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7
    input_tensor = np.array([0, 1000, 2000], dtype=np.int32)
    input_min = np.array(-3000.0, dtype=np.float32)
    input_max = np.array(3000.0, dtype=np.float32)
    requested_output_min = np.array(-100.0, dtype=np.float32)
    requested_output_max = np.array(100.0, dtype=np.float32)
    out_type = tf.qint8

    input_dict = {
        "input": input_tensor,
        "input_min": input_min,
        "input_max": input_max,
        "requested_output_min": requested_output_min,
        "requested_output_max": requested_output_max,
        "out_type": out_type,
        "name": "RequantizeExample7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array([0, 100, 200], dtype=np.uint8)
    input_min = np.array(0.0, dtype=np.float32)
    input_max = np.array(255.0, dtype=np.float32)
    requested_output_min = np.array(0.0, dtype=np.float32)
    requested_output_max = np.array(127.0, dtype=np.float32)
    out_type = tf.qint8

    input_dict = {
        "input": input_tensor,
        "input_min": input_min,
        "input_max": input_max,
        "requested_output_min": requested_output_min,
        "requested_output_max": requested_output_max,
        "out_type": out_type,
        "name": "RequantizeExample8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.array([0, 32767], dtype=np.int16)
    input_min = np.array(-32768.0, dtype=np.float32)
    input_max = np.array(32767.0, dtype=np.float32)
    requested_output_min = np.array(-128.0, dtype=np.float32)
    requested_output_max = np.array(127.0, dtype=np.float32)
    out_type = tf.qint8

    input_dict = {
        "input": input_tensor,
        "input_min": input_min,
        "input_max": input_max,
        "requested_output_min": requested_output_min,
        "requested_output_max": requested_output_max,
        "out_type": out_type,
        "name": "RequantizeExample9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.array([0, 65000], dtype=np.uint16)
    input_min = np.array(0.0, dtype=np.float32)
    input_max = np.array(65535.0, dtype=np.float32)
    requested_output_min = np.array(0.0, dtype=np.float32)
    requested_output_max = np.array(255.0, dtype=np.float32)
    out_type = tf.quint8

    input_dict = {
        "input": input_tensor,
        "input_min": input_min,
        "input_max": input_max,
        "requested_output_min": requested_output_min,
        "requested_output_max": requested_output_max,
        "out_type": out_type,
        "name": "RequantizeExample10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
inputs = tf_raw_ops_requantize_inputs()
generated_inputs["tf.raw_ops.Requantize"] = []
for input_dict in inputs:
    generated_inputs["tf.raw_ops.Requantize"].append(input_dict)

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Requantize' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Requantize'.")

check_valid('tf.raw_ops.Requantize', generated_inputs['tf.raw_ops.Requantize'], lib="tf", suffix=0)
