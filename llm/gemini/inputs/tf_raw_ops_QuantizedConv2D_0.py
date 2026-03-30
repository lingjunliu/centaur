
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_QuantizedConv2D_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([[[[1, 2]], [[3, 4]]]], dtype=np.qint8)
    filter_tensor = np.array([[[[1]], [[2]]]], dtype=np.qint8)
    min_input_tensor = np.float32(0.0)
    max_input_tensor = np.float32(5.0)
    min_filter_tensor = np.float32(-1.0)
    max_filter_tensor = np.float32(2.0)
    strides_val = [1, 1, 1, 1]
    padding_val = "VALID"
    out_type_val = tf.qint32
    dilations_val = [1, 1, 1, 1]
    name_val = "conv1"

    input_dict = {
        "input": input_tensor,
        "filter": filter_tensor,
        "min_input": min_input_tensor,
        "max_input": max_input_tensor,
        "min_filter": min_filter_tensor,
        "max_filter": max_filter_tensor,
        "strides": strides_val,
        "padding": padding_val,
        "out_type": out_type_val,
        "dilations": dilations_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[[[1, 2, 3], [4, 5, 6]]]], dtype=np.quint8)
    filter_tensor = np.array([[[[1], [2]], [[3], [4]]]], dtype=np.quint8)
    min_input_tensor = np.float32(0.0)
    max_input_tensor = np.float32(255.0)
    min_filter_tensor = np.float32(0.0)
    max_filter_tensor = np.float32(255.0)
    strides_val = [1, 2, 2, 1]
    padding_val = "SAME"
    out_type_val = tf.qint16
    dilations_val = [1, 1, 1, 1]
    name_val = "conv2"

    input_dict = {
        "input": input_tensor,
        "filter": filter_tensor,
        "min_input": min_input_tensor,
        "max_input": max_input_tensor,
        "min_filter": min_filter_tensor,
        "max_filter": max_filter_tensor,
        "strides": strides_val,
        "padding": padding_val,
        "out_type": out_type_val,
        "dilations": dilations_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([[[[[1, 2], [3, 4]]]], [[[[5, 6], [7, 8]]]]], dtype=np.qint8)
    filter_tensor = np.array([[[[[1], [2]]]], [[[[3], [4]]]]], dtype=np.qint8)
    min_input_tensor = np.float32(-10.0)
    max_input_tensor = np.float32(10.0)
    min_filter_tensor = np.float32(-5.0)
    max_filter_tensor = np.float32(5.0)
    strides_val = [1, 1, 1, 1]
    padding_val = "VALID"
    out_type_val = tf.quint8
    dilations_val = [1, 2, 2, 1]
    name_val = "conv3"

    input_dict = {
        "input": input_tensor,
        "filter": filter_tensor,
        "min_input": min_input_tensor,
        "max_input": max_input_tensor,
        "min_filter": min_filter_tensor,
        "max_filter": max_filter_tensor,
        "strides": strides_val,
        "padding": padding_val,
        "out_type": out_type_val,
        "dilations": dilations_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([[[[1, 2, 3], [4, 5, 6]]]], dtype=np.quint16)
    filter_tensor = np.array([[[[1], [2]], [[3], [4]]]], dtype=np.quint16)
    min_input_tensor = np.float32(0.0)
    max_input_tensor = np.float32(65535.0)
    min_filter_tensor = np.float32(0.0)
    max_filter_tensor = np.float32(65535.0)
    strides_val = [1, 2, 2, 1]
    padding_val = "SAME"
    out_type_val = tf.qint32
    dilations_val = [1, 1, 1, 1]
    name_val = "conv4"

    input_dict = {
        "input": input_tensor,
        "filter": filter_tensor,
        "min_input": min_input_tensor,
        "max_input": max_input_tensor,
        "min_filter": min_filter_tensor,
        "max_filter": max_filter_tensor,
        "strides": strides_val,
        "padding": padding_val,
        "out_type": out_type_val,
        "dilations": dilations_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([[[[[1, 2], [3, 4]]]], [[[[5, 6], [7, 8]]]]], dtype=np.qint32)
    filter_tensor = np.array([[[[[1], [2]]]], [[[[3], [4]]]]], dtype=np.qint32)
    min_input_tensor = np.float32(-100.0)
    max_input_tensor = np.float32(100.0)
    min_filter_tensor = np.float32(-50.0)
    max_filter_tensor = np.float32(50.0)
    strides_val = [1, 1, 1, 1]
    padding_val = "VALID"
    out_type_val = tf.quint16
    dilations_val = [1, 3, 3, 1]
    name_val = "conv5"

    input_dict = {
        "input": input_tensor,
        "filter": filter_tensor,
        "min_input": min_input_tensor,
        "max_input": max_input_tensor,
        "min_filter": min_filter_tensor,
        "max_filter": max_filter_tensor,
        "strides": strides_val,
        "padding": padding_val,
        "out_type": out_type_val,
        "dilations": dilations_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array([[[[1, 2]], [[3, 4]]]], dtype=np.qint16)
    filter_tensor = np.array([[[[1]], [[2]]]], dtype=np.qint16)
    min_input_tensor = np.float32(-10.0)
    max_input_tensor = np.float32(10.0)
    min_filter_tensor = np.float32(-5.0)
    max_filter_tensor = np.float32(5.0)
    strides_val = [1, 1, 1, 1]
    padding_val = "VALID"
    out_type_val = tf.qint8
    dilations_val = [1, 1, 1, 1]
    name_val = "conv6"

    input_dict = {
        "input": input_tensor,
        "filter": filter_tensor,
        "min_input": min_input_tensor,
        "max_input": max_input_tensor,
        "min_filter": min_filter_tensor,
        "max_filter": max_filter_tensor,
        "strides": strides_val,
        "padding": padding_val,
        "out_type": out_type_val,
        "dilations": dilations_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array([[[[1, 2, 3], [4, 5, 6]]]], dtype=np.qint8)
    filter_tensor = np.array([[[[1], [2]], [[3], [4]]]], dtype=np.qint8)
    min_input_tensor = np.float32(-128.0)
    max_input_tensor = np.float32(127.0)
    min_filter_tensor = np.float32(-128.0)
    max_filter_tensor = np.float32(127.0)
    strides_val = [1, 2, 2, 1]
    padding_val = "SAME"
    out_type_val = tf.qint32
    dilations_val = [1, 1, 1, 1]
    name_val = "conv7"

    input_dict = {
        "input": input_tensor,
        "filter": filter_tensor,
        "min_input": min_input_tensor,
        "max_input": max_input_tensor,
        "min_filter": min_filter_tensor,
        "max_filter": max_filter_tensor,
        "strides": strides_val,
        "padding": padding_val,
        "out_type": out_type_val,
        "dilations": dilations_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array([[[[[1, 2], [3, 4]]]], [[[[5, 6], [7, 8]]]]], dtype=np.quint8)
    filter_tensor = np.array([[[[[1], [2]]]], [[[[3], [4]]]]], dtype=np.quint8)
    min_input_tensor = np.float32(0.0)
    max_input_tensor = np.float32(255.0)
    min_filter_tensor = np.float32(0.0)
    max_filter_tensor = np.float32(255.0)
    strides_val = [1, 1, 1, 1]
    padding_val = "VALID"
    out_type_val = tf.qint16
    dilations_val = [1, 2, 2, 1]
    name_val = "conv8"

    input_dict = {
        "input": input_tensor,
        "filter": filter_tensor,
        "min_input": min_input_tensor,
        "max_input": max_input_tensor,
        "min_filter": min_filter_tensor,
        "max_filter": max_filter_tensor,
        "strides": strides_val,
        "padding": padding_val,
        "out_type": out_type_val,
        "dilations": dilations_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 9
    input_tensor = np.array([[[[[1, 2], [3, 4]]]], [[[[5, 6], [7, 8]]]]], dtype=np.qint32)
    filter_tensor = np.array([[[[[1], [2]]]], [[[[3], [4]]]]], dtype=np.qint32)
    min_input_tensor = np.float32(-2147483648.0)
    max_input_tensor = np.float32(2147483647.0)
    min_filter_tensor = np.float32(-2147483648.0)
    max_filter_tensor = np.float32(2147483647.0)
    strides_val = [1, 1, 1, 1]
    padding_val = "VALID"
    out_type_val = tf.quint16
    dilations_val = [1, 3, 3, 1]
    name_val = "conv9"

    input_dict = {
        "input": input_tensor,
        "filter": filter_tensor,
        "min_input": min_input_tensor,
        "max_input": max_input_tensor,
        "min_filter": min_filter_tensor,
        "max_filter": max_filter_tensor,
        "strides": strides_val,
        "padding": padding_val,
        "out_type": out_type_val,
        "dilations": dilations_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.array([[[[1, 2]], [[3, 4]]]], dtype=np.qint8)
    filter_tensor = np.array([[[[1]], [[2]]]], dtype=np.qint8)
    min_input_tensor = np.float32(0.0)
    max_input_tensor = np.float32(5.0)
    min_filter_tensor = np.float32(-1.0)
    max_filter_tensor = np.float32(2.0)
    strides_val = [1, 1, 1, 1]
    padding_val = "VALID"
    out_type_val = tf.qint32
    dilations_val = [1, 1, 1, 1]
    name_val = "conv10"

    input_dict = {
        "input": input_tensor,
        "filter": filter_tensor,
        "min_input": min_input_tensor,
        "max_input": max_input_tensor,
        "min_filter": min_filter_tensor,
        "max_filter": max_filter_tensor,
        "strides": strides_val,
        "padding": padding_val,
        "out_type": out_type_val,
        "dilations": dilations_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.QuantizedConv2D"] = tf_raw_ops_QuantizedConv2D_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.QuantizedConv2D' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QuantizedConv2D'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.QuantizedConv2D', generated_inputs['tf.raw_ops.QuantizedConv2D'], lib="tf", suffix=0)
