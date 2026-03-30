
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_requantize_inputs():
    list_of_inputs = []

    # Input 1: Basic case with qint8
    input_data = np.array([10, 20, 30, 40], dtype=np.int8)
    input_min = np.array(-1.0, dtype=np.float32)
    input_max = np.array(1.0, dtype=np.float32)
    requested_output_min = np.array(-0.5, dtype=np.float32)
    requested_output_max = np.array(0.5, dtype=np.float32)
    out_type = tf.qint8

    input_dict = {
        "input": input_data,
        "input_min": input_min,
        "input_max": input_max,
        "requested_output_min": requested_output_min,
        "requested_output_max": requested_output_max,
        "out_type": out_type,
        "name": "requantize_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: quint8 with different ranges
    input_data = np.array([50, 100, 150, 200], dtype=np.uint8)
    input_min = np.array(0.0, dtype=np.float32)
    input_max = np.array(255.0, dtype=np.float32)
    requested_output_min = np.array(0.0, dtype=np.float32)
    requested_output_max = np.array(100.0, dtype=np.float32)
    out_type = tf.qint8

    input_dict = {
        "input": input_data,
        "input_min": input_min,
        "input_max": input_max,
        "requested_output_min": requested_output_min,
        "requested_output_max": requested_output_max,
        "out_type": out_type,
        "name": "requantize_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: qint32 with negative values
    input_data = np.array([-10000, 0, 10000, 20000], dtype=np.int32)
    input_min = np.array(-30000.0, dtype=np.float32)
    input_max = np.array(30000.0, dtype=np.float32)
    requested_output_min = np.array(-10.0, dtype=np.float32)
    requested_output_max = np.array(10.0, dtype=np.float32)
    out_type = tf.qint32

    input_dict = {
        "input": input_data,
        "input_min": input_min,
        "input_max": input_max,
        "requested_output_min": requested_output_min,
        "requested_output_max": requested_output_max,
        "out_type": out_type,
        "name": "requantize_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: qint16 with 2D input
    input_data = np.array([[100, 200], [300, 400]], dtype=np.int16)
    input_min = np.array(0.0, dtype=np.float32)
    input_max = np.array(500.0, dtype=np.float32)
    requested_output_min = np.array(0.0, dtype=np.float32)
    requested_output_max = np.array(1.0, dtype=np.float32)
    out_type = tf.qint16

    input_dict = {
        "input": input_data,
        "input_min": input_min,
        "input_max": input_max,
        "requested_output_min": requested_output_min,
        "requested_output_max": requested_output_max,
        "out_type": out_type,
        "name": "requantize_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 5: quint16 with different range
    input_data = np.array([1000, 2000, 3000, 4000], dtype=np.uint16)
    input_min = np.array(0.0, dtype=np.float32)
    input_max = np.array(5000.0, dtype=np.float32)
    requested_output_min = np.array(0.0, dtype=np.float32)
    requested_output_max = np.array(200.0, dtype=np.float32)
    out_type = tf.quint16

    input_dict = {
        "input": input_data,
        "input_min": input_min,
        "input_max": input_max,
        "requested_output_min": requested_output_min,
        "requested_output_max": requested_output_max,
        "out_type": out_type,
        "name": "requantize_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D input qint8
    input_data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int8)
    input_min = np.array(-10.0, dtype=np.float32)
    input_max = np.array(10.0, dtype=np.float32)
    requested_output_min = np.array(-5.0, dtype=np.float32)
    requested_output_max = np.array(5.0, dtype=np.float32)
    out_type = tf.qint8

    input_dict = {
        "input": input_data,
        "input_min": input_min,
        "input_max": input_max,
        "requested_output_min": requested_output_min,
        "requested_output_max": requested_output_max,
        "out_type": out_type,
        "name": "requantize_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large values for qint32
    input_data = np.array([-2000000, 0, 2000000, 4000000], dtype=np.int32)
    input_min = np.array(-5000000.0, dtype=np.float32)
    input_max = np.array(5000000.0, dtype=np.float32)
    requested_output_min = np.array(-100.0, dtype=np.float32)
    requested_output_max = np.array(100.0, dtype=np.float32)
    out_type = tf.qint32

    input_dict = {
        "input": input_data,
        "input_min": input_min,
        "input_max": input_max,
        "requested_output_min": requested_output_min,
        "requested_output_max": requested_output_max,
        "out_type": out_type,
        "name": "requantize_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Edge case qint8
    input_data = np.array([-128, -1, 0, 1, 127], dtype=np.int8)
    input_min = np.array(-128.0, dtype=np.float32)
    input_max = np.array(127.0, dtype=np.float32)
    requested_output_min = np.array(-10.0, dtype=np.float32)
    requested_output_max = np.array(10.0, dtype=np.float32)
    out_type = tf.qint8

    input_dict = {
        "input": input_data,
        "input_min": input_min,
        "input_max": input_max,
        "requested_output_min": requested_output_min,
        "requested_output_max": requested_output_max,
        "out_type": out_type,
        "name": "requantize_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Edge case quint8
    input_data = np.array([0, 1, 2, 254, 255], dtype=np.uint8)
    input_min = np.array(0.0, dtype=np.float32)
    input_max = np.array(255.0, dtype=np.float32)
    requested_output_min = np.array(0.0, dtype=np.float32)
    requested_output_max = np.array(5.0, dtype=np.float32)
    out_type = tf.quint8

    input_dict = {
        "input": input_data,
        "input_min": input_min,
        "input_max": input_max,
        "requested_output_min": requested_output_min,
        "requested_output_max": requested_output_max,
        "out_type": out_type,
        "name": "requantize_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: With different dimensions.
    input_data = np.array([[[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]],[[[13,14,15],[16,17,18]],[[19,20,21],[22,23,24]]]], dtype=np.int8)
    input_min = np.array(-1.0, dtype=np.float32)
    input_max = np.array(1.0, dtype=np.float32)
    requested_output_min = np.array(-0.5, dtype=np.float32)
    requested_output_max = np.array(0.5, dtype=np.float32)
    out_type = tf.qint8

    input_dict = {
        "input": input_data,
        "input_min": input_min,
        "input_max": input_max,
        "requested_output_min": requested_output_min,
        "requested_output_max": requested_output_max,
        "out_type": out_type,
        "name": "requantize_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Requantize"] = tf_raw_ops_requantize_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Requantize' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Requantize'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Requantize', generated_inputs['tf.raw_ops.Requantize'], lib="tf", suffix=0)
