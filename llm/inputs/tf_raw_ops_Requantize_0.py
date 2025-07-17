
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_Requantize_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int8)
    input_min_tensor = np.array(-1.0, dtype=np.float32)
    input_max_tensor = np.array(1.0, dtype=np.float32)
    requested_output_min_tensor = np.array(-0.5, dtype=np.float32)
    requested_output_max_tensor = np.array(0.5, dtype=np.float32)
    out_type = tf.qint8
    #input_tensor = tf.quantization.quantize(input_tensor, input_min_tensor, input_max_tensor, tf.qint8)[0]
    input_tensor = tf.constant(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int8))


    input_dict = {
        "input": input_tensor,
        "input_min": input_min_tensor,
        "input_max": input_max_tensor,
        "requested_output_min": requested_output_min_tensor,
        "requested_output_max": requested_output_max_tensor,
        "out_type": out_type,
        "name": "requantize_op_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[-1, 0, 1], [2, 3, 4]], dtype=np.int8)
    input_min_tensor = np.array(-2.0, dtype=np.float32)
    input_max_tensor = np.array(4.0, dtype=np.float32)
    requested_output_min_tensor = np.array(-1.0, dtype=np.float32)
    requested_output_max_tensor = np.array(2.0, dtype=np.float32)
    out_type = tf.quint8
    #input_tensor = tf.quantization.quantize(input_tensor, input_min_tensor, input_max_tensor, tf.quint8)[0]
    input_tensor = tf.constant(np.array([[-1, 0, 1], [2, 3, 4]], dtype=np.int8))


    input_dict = {
        "input": input_tensor,
        "input_min": input_min_tensor,
        "input_max": input_max_tensor,
        "requested_output_min": requested_output_min_tensor,
        "requested_output_max": requested_output_max_tensor,
        "out_type": out_type,
        "name": "requantize_op_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([100, 200, 300], dtype=np.int32)
    input_min_tensor = np.array(0.0, dtype=np.float32)
    input_max_tensor = np.array(500.0, dtype=np.float32)
    requested_output_min_tensor = np.array(0.0, dtype=np.float32)
    requested_output_max_tensor = np.array(250.0, dtype=np.float32)
    out_type = tf.qint16

    #input_tensor = tf.quantization.quantize(input_tensor, input_min_tensor, input_max_tensor, tf.qint32)[0]
    input_tensor = tf.constant(np.array([100, 200, 300], dtype=np.int32))


    input_dict = {
        "input": input_tensor,
        "input_min": input_min_tensor,
        "input_max": input_max_tensor,
        "requested_output_min": requested_output_min_tensor,
        "requested_output_max": requested_output_max_tensor,
        "out_type": out_type,
        "name": "requantize_op_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
    input_min_tensor = np.array(0.0, dtype=np.float32)
    input_max_tensor = np.array(255.0, dtype=np.float32)
    requested_output_min_tensor = np.array(0.0, dtype=np.float32)
    requested_output_max_tensor = np.array(127.0, dtype=np.float32)
    out_type = tf.qint8

    #input_tensor = tf.quantization.quantize(input_tensor, input_min_tensor, input_max_tensor, tf.quint8)[0]
    input_tensor = tf.constant(np.array([1, 2, 3, 4, 5], dtype=np.uint8))


    input_dict = {
        "input": input_tensor,
        "input_min": input_min_tensor,
        "input_max": input_max_tensor,
        "requested_output_min": requested_output_min_tensor,
        "requested_output_max": requested_output_max_tensor,
        "out_type": out_type,
        "name": "requantize_op_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 5
    input_tensor = np.array([10, 20, 30], dtype=np.int16)
    input_min_tensor = np.array(-100.0, dtype=np.float32)
    input_max_tensor = np.array(100.0, dtype=np.float32)
    requested_output_min_tensor = np.array(-50.0, dtype=np.float32)
    requested_output_max_tensor = np.array(50.0, dtype=np.float32)
    out_type = tf.qint8
    #input_tensor = tf.quantization.quantize(input_tensor, input_min_tensor, input_max_tensor, tf.qint16)[0]
    input_tensor = tf.constant(np.array([10, 20, 30], dtype=np.int16))


    input_dict = {
        "input": input_tensor,
        "input_min": input_min_tensor,
        "input_max": input_max_tensor,
        "requested_output_min": requested_output_min_tensor,
        "requested_output_max": requested_output_max_tensor,
        "out_type": out_type,
        "name": "requantize_op_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint16).reshape((2, 4))
    input_min_tensor = np.array(0.0, dtype=np.float32)
    input_max_tensor = np.array(65535.0, dtype=np.float32)
    requested_output_min_tensor = np.array(0.0, dtype=np.float32)
    requested_output_max_tensor = np.array(255.0, dtype=np.float32)
    out_type = tf.quint8

    #input_tensor = tf.quantization.quantize(input_tensor, input_min_tensor, input_max_tensor, tf.quint16)[0]
    input_tensor = tf.constant(np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint16).reshape((2, 4)))


    input_dict = {
        "input": input_tensor,
        "input_min": input_min_tensor,
        "input_max": input_max_tensor,
        "requested_output_min": requested_output_min_tensor,
        "requested_output_max": requested_output_max_tensor,
        "out_type": out_type,
        "name": "requantize_op_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array([-1000, -500, 0, 500, 1000], dtype=np.int32)
    input_min_tensor = np.array(-2000.0, dtype=np.float32)
    input_max_tensor = np.array(2000.0, dtype=np.float32)
    requested_output_min_tensor = np.array(-1000.0, dtype=np.float32)
    requested_output_max_tensor = np.array(1000.0, dtype=np.float32)
    out_type = tf.qint16

    #input_tensor = tf.quantization.quantize(input_tensor, input_min_tensor, input_max_tensor, tf.qint32)[0]
    input_tensor = tf.constant(np.array([-1000, -500, 0, 500, 1000], dtype=np.int32))


    input_dict = {
        "input": input_tensor,
        "input_min": input_min_tensor,
        "input_max": input_max_tensor,
        "requested_output_min": requested_output_min_tensor,
        "requested_output_max": requested_output_max_tensor,
        "out_type": out_type,
        "name": "requantize_op_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array([255, 128, 0], dtype=np.uint8)
    input_min_tensor = np.array(0.0, dtype=np.float32)
    input_max_tensor = np.array(255.0, dtype=np.float32)
    requested_output_min_tensor = np.array(0.0, dtype=np.float32)
    requested_output_max_tensor = np.array(63.0, dtype=np.float32)
    out_type = tf.qint8

    #input_tensor = tf.quantization.quantize(input_tensor, input_min_tensor, input_max_tensor, tf.quint8)[0]
    input_tensor = tf.constant(np.array([255, 128, 0], dtype=np.uint8))


    input_dict = {
        "input": input_tensor,
        "input_min": input_min_tensor,
        "input_max": input_max_tensor,
        "requested_output_min": requested_output_min_tensor,
        "requested_output_max": requested_output_max_tensor,
        "out_type": out_type,
        "name": "requantize_op_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.array([[-32768, 0, 32767], [16384, -16384, 0]], dtype=np.int16)
    input_min_tensor = np.array(-32768.0, dtype=np.float32)
    input_max_tensor = np.array(32767.0, dtype=np.float32)
    requested_output_min_tensor = np.array(-100.0, dtype=np.float32)
    requested_output_max_tensor = np.array(100.0, dtype=np.float32)
    out_type = tf.qint8

    #input_tensor = tf.quantization.quantize(input_tensor, input_min_tensor, input_max_tensor, tf.qint16)[0]
    input_tensor = tf.constant(np.array([[-32768, 0, 32767], [16384, -16384, 0]], dtype=np.int16))


    input_dict = {
        "input": input_tensor,
        "input_min": input_min_tensor,
        "input_max": input_max_tensor,
        "requested_output_min": requested_output_min_tensor,
        "requested_output_max": requested_output_max_tensor,
        "out_type": out_type,
        "name": "requantize_op_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.array([0, 32767, 65535], dtype=np.uint16)
    input_min_tensor = np.array(0.0, dtype=np.float32)
    input_max_tensor = np.array(65535.0, dtype=np.float32)
    requested_output_min_tensor = np.array(0.0, dtype=np.float32)
    requested_output_max_tensor = np.array(255.0, dtype=np.float32)
    out_type = tf.qint8

    #input_tensor = tf.quantization.quantize(input_tensor, input_min_tensor, input_max_tensor, tf.quint16)[0]
    input_tensor = tf.constant(np.array([0, 32767, 65535], dtype=np.uint16))


    input_dict = {
        "input": input_tensor,
        "input_min": input_min_tensor,
        "input_max": input_max_tensor,
        "requested_output_min": requested_output_min_tensor,
        "requested_output_max": requested_output_max_tensor,
        "out_type": out_type,
        "name": "requantize_op_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Requantize"] = tf_raw_ops_Requantize_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Requantize' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Requantize'.")

check_valid('tf.raw_ops.Requantize', generated_inputs['tf.raw_ops.Requantize'], lib="tf", suffix=0)
