
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_QuantizedConv2D_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([[[[1], [2]], [[3], [4]]]], dtype=np.int8)
    filter_tensor = np.array([[[[1]], [[2]]]], dtype=np.int8)
    min_input_val = np.array(-1.0, dtype=np.float32)
    max_input_val = np.array(5.0, dtype=np.float32)
    min_filter_val = np.array(-2.0, dtype=np.float32)
    max_filter_val = np.array(2.0, dtype=np.float32)
    strides_val = [1, 1, 1, 1]
    padding_val = "VALID"
    out_type_val = tf.qint32
    dilations_val = [1, 1, 1, 1]
    name_val = "conv2d_1"

    input_dict = {
        "input": input_tensor,
        "filter": filter_tensor,
        "min_input": min_input_val,
        "max_input": max_input_val,
        "min_filter": min_filter_val,
        "max_filter": max_filter_val,
        "strides": strides_val,
        "padding": padding_val,
        "out_type": out_type_val,
        "dilations": dilations_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[[[1, 2], [2, 3]], [[3, 4], [4, 5]]]], dtype=np.uint8)
    filter_tensor = np.array([[[[1, 2]], [[2, 3]]]], dtype=np.uint8)
    min_input_val = np.array(0.0, dtype=np.float32)
    max_input_val = np.array(255.0, dtype=np.float32)
    min_filter_val = np.array(0.0, dtype=np.float32)
    max_filter_val = np.array(255.0, dtype=np.float32)
    strides_val = [1, 1, 1, 1]
    padding_val = "SAME"
    out_type_val = tf.qint8
    dilations_val = [1, 1, 1, 1]
    name_val = "conv2d_2"
    input_dict = {
        "input": input_tensor,
        "filter": filter_tensor,
        "min_input": min_input_val,
        "max_input": max_input_val,
        "min_filter": min_filter_val,
        "max_filter": max_filter_val,
        "strides": strides_val,
        "padding": padding_val,
        "out_type": out_type_val,
        "dilations": dilations_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([[[[[1], [2]], [[3], [4]]], [[[5], [6]], [[7], [8]]]]], dtype=np.int16)
    filter_tensor = np.array([[[[[1]], [[2]]]], [[[[3]], [[4]]]]], dtype=np.int16)
    min_input_val = np.array(-32768.0, dtype=np.float32)
    max_input_val = np.array(32767.0, dtype=np.float32)
    min_filter_val = np.array(-32768.0, dtype=np.float32)
    max_filter_val = np.array(32767.0, dtype=np.float32)
    strides_val = [1, 1, 2, 1]
    padding_val = "VALID"
    out_type_val = tf.qint16
    dilations_val = [1, 1, 1, 1]
    name_val = None
    input_dict = {
        "input": input_tensor,
        "filter": filter_tensor,
        "min_input": min_input_val,
        "max_input": max_input_val,
        "min_filter": min_filter_val,
        "max_filter": max_filter_val,
        "strides": strides_val,
        "padding": padding_val,
        "out_type": out_type_val,
        "dilations": dilations_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([[[[1, 2, 3], [2, 3, 4]], [[3, 4, 5], [4, 5, 6]]]], dtype=np.uint16)
    filter_tensor = np.array([[[[1, 2, 3]], [[2, 3, 4]]]], dtype=np.uint16)
    min_input_val = np.array(10.0, dtype=np.float32)
    max_input_val = np.array(20.0, dtype=np.float32)
    min_filter_val = np.array(5.0, dtype=np.float32)
    max_filter_val = np.array(15.0, dtype=np.float32)
    strides_val = [1, 2, 1, 1]
    padding_val = "SAME"
    out_type_val = tf.qint16
    dilations_val = [1, 1, 1, 1]
    name_val = "conv2d_4"
    input_dict = {
        "input": input_tensor,
        "filter": filter_tensor,
        "min_input": min_input_val,
        "max_input": max_input_val,
        "min_filter": min_filter_val,
        "max_filter": max_filter_val,
        "strides": strides_val,
        "padding": padding_val,
        "out_type": out_type_val,
        "dilations": dilations_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 5
    input_tensor = np.array([[[[1], [2]], [[3], [4]]]], dtype=np.int32)
    filter_tensor = np.array([[[[1]], [[2]]]], dtype=np.int32)
    min_input_val = np.array(-100.0, dtype=np.float32)
    max_input_val = np.array(100.0, dtype=np.float32)
    min_filter_val = np.array(-50.0, dtype=np.float32)
    max_filter_val = np.array(50.0, dtype=np.float32)
    strides_val = [1, 1, 1, 1]
    padding_val = "VALID"
    out_type_val = tf.qint16
    dilations_val = [1, 1, 1, 1]
    name_val = None
    input_dict = {
        "input": input_tensor,
        "filter": filter_tensor,
        "min_input": min_input_val,
        "max_input": max_input_val,
        "min_filter": min_filter_val,
        "max_filter": max_filter_val,
        "strides": strides_val,
        "padding": padding_val,
        "out_type": out_type_val,
        "dilations": dilations_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array([[[[1, 2], [2, 3]], [[3, 4], [4, 5]]]], dtype=np.int8)
    filter_tensor = np.array([[[[1, 2]], [[2, 3]]]], dtype=np.int8)
    min_input_val = np.array(-5.0, dtype=np.float32)
    max_input_val = np.array(10.0, dtype=np.float32)
    min_filter_val = np.array(-2.0, dtype=np.float32)
    max_filter_val = np.array(5.0, dtype=np.float32)
    strides_val = [1, 2, 2, 1]
    padding_val = "SAME"
    out_type_val = tf.qint8
    dilations_val = [1, 1, 1, 1]
    name_val = "conv2d_6"
    input_dict = {
        "input": input_tensor,
        "filter": filter_tensor,
        "min_input": min_input_val,
        "max_input": max_input_val,
        "min_filter": min_filter_val,
        "max_filter": max_filter_val,
        "strides": strides_val,
        "padding": padding_val,
        "out_type": out_type_val,
        "dilations": dilations_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array([[[[[1], [2]], [[3], [4]]], [[[5], [6]], [[7], [8]]]]], dtype=np.uint8)
    filter_tensor = np.array([[[[[1]], [[2]]]], [[[[3]], [[4]]]]], dtype=np.uint8)
    min_input_val = np.array(0.0, dtype=np.float32)
    max_input_val = np.array(100.0, dtype=np.float32)
    min_filter_val = np.array(0.0, dtype=np.float32)
    max_filter_val = np.array(50.0, dtype=np.float32)
    strides_val = [1, 1, 1, 1]
    padding_val = "VALID"
    out_type_val = tf.qint8
    dilations_val = [1, 1, 1, 1]
    name_val = None
    input_dict = {
        "input": input_tensor,
        "filter": filter_tensor,
        "min_input": min_input_val,
        "max_input": max_input_val,
        "min_filter": min_filter_val,
        "max_filter": max_filter_val,
        "strides": strides_val,
        "padding": padding_val,
        "out_type": out_type_val,
        "dilations": dilations_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array([[[[1, 2, 3], [2, 3, 4]], [[3, 4, 5], [4, 5, 6]]]], dtype=np.int16)
    filter_tensor = np.array([[[[1, 2, 3]], [[2, 3, 4]]]], dtype=np.int16)
    min_input_val = np.array(-100.0, dtype=np.float32)
    max_input_val = np.array(200.0, dtype=np.float32)
    min_filter_val = np.array(-50.0, dtype=np.float32)
    max_filter_val = np.array(100.0, dtype=np.float32)
    strides_val = [1, 2, 1, 1]
    padding_val = "SAME"
    out_type_val = tf.qint16
    dilations_val = [1, 1, 1, 1]
    name_val = "conv2d_8"
    input_dict = {
        "input": input_tensor,
        "filter": filter_tensor,
        "min_input": min_input_val,
        "max_input": max_input_val,
        "min_filter": min_filter_val,
        "max_filter": max_filter_val,
        "strides": strides_val,
        "padding": padding_val,
        "out_type": out_type_val,
        "dilations": dilations_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.array([[[[1], [2]], [[3], [4]]]], dtype=np.uint16)
    filter_tensor = np.array([[[[1]], [[2]]]], dtype=np.uint16)
    min_input_val = np.array(0.0, dtype=np.float32)
    max_input_val = np.array(65535.0, dtype=np.float32)
    min_filter_val = np.array(0.0, dtype=np.float32)
    max_filter_val = np.array(65535.0, dtype=np.float32)
    strides_val = [1, 1, 1, 1]
    padding_val = "VALID"
    out_type_val = tf.qint16
    dilations_val = [1, 1, 1, 1]
    name_val = None
    input_dict = {
        "input": input_tensor,
        "filter": filter_tensor,
        "min_input": min_input_val,
        "max_input": max_input_val,
        "min_filter": min_filter_val,
        "max_filter": max_filter_val,
        "strides": strides_val,
        "padding": padding_val,
        "out_type": out_type_val,
        "dilations": dilations_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.array([[[[1, 2], [2, 3]], [[3, 4], [4, 5]]]], dtype=np.int32)
    filter_tensor = np.array([[[[1, 2]], [[2, 3]]]], dtype=np.int32)
    min_input_val = np.array(-2147483648.0, dtype=np.float32)
    max_input_val = np.array(2147483647.0, dtype=np.float32)
    min_filter_val = np.array(-2147483648.0, dtype=np.float32)
    max_filter_val = np.array(2147483647.0, dtype=np.float32)
    strides_val = [1, 1, 1, 1]
    padding_val = "SAME"
    out_type_val = tf.qint16
    dilations_val = [1, 1, 1, 1]
    name_val = "conv2d_10"
    input_dict = {
        "input": input_tensor,
        "filter": filter_tensor,
        "min_input": min_input_val,
        "max_input": max_input_val,
        "min_filter": min_filter_val,
        "max_filter": max_filter_val,
        "strides": strides_val,
        "padding": padding_val,
        "out_type": out_type_val,
        "dilations": dilations_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.QuantizedConv2D"] = tf_raw_ops_QuantizedConv2D_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.QuantizedConv2D' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QuantizedConv2D'.")

check_valid('tf.raw_ops.QuantizedConv2D', generated_inputs['tf.raw_ops.QuantizedConv2D'], lib="tf", suffix=0)
