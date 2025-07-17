
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_QuantizedConv2D_inputs():
    list_of_inputs = []

    # Input 1
    input1 = np.array([[[[1], [2]], [[3], [4]]]], dtype=np.int8)
    filter1 = np.array([[[[1]], [[2]]]], dtype=np.int8)
    min_input1 = np.array([0.0], dtype=np.float32)
    max_input1 = np.array([5.0], dtype=np.float32)
    min_filter1 = np.array([0.0], dtype=np.float32)
    max_filter1 = np.array([3.0], dtype=np.float32)
    strides1 = [1, 1, 1, 1]
    padding1 = "VALID"
    out_type1 = tf.qint32
    dilations1 = [1, 1, 1, 1]
    name1 = "conv1"

    input_dict1 = {
        "input": tf.constant(input1, dtype=tf.qint8),
        "filter": tf.constant(filter1, dtype=tf.qint8),
        "min_input": tf.constant(min_input1, dtype=tf.float32),
        "max_input": tf.constant(max_input1, dtype=tf.float32),
        "min_filter": tf.constant(min_filter1, dtype=tf.float32),
        "max_filter": tf.constant(max_filter1, dtype=tf.float32),
        "strides": strides1,
        "padding": padding1,
        "out_type": out_type1,
        "dilations": dilations1,
        "name": name1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2
    input2 = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.uint8)
    filter2 = np.array([[[[1, 2]], [[3, 4]]]], dtype=np.uint8)
    min_input2 = np.array([0.0], dtype=np.float32)
    max_input2 = np.array([255.0], dtype=np.float32)
    min_filter2 = np.array([0.0], dtype=np.float32)
    max_filter2 = np.array([255.0], dtype=np.float32)
    strides2 = [1, 1, 1, 1]
    padding2 = "SAME"
    out_type2 = tf.qint32
    dilations2 = [1, 1, 1, 1]
    name2 = "conv2"

    input_dict2 = {
        "input": tf.constant(input2, dtype=tf.quint8),
        "filter": tf.constant(filter2, dtype=tf.quint8),
        "min_input": tf.constant(min_input2, dtype=tf.float32),
        "max_input": tf.constant(max_input2, dtype=tf.float32),
        "min_filter": tf.constant(min_filter2, dtype=tf.float32),
        "max_filter": tf.constant(max_filter2, dtype=tf.float32),
        "strides": strides2,
        "padding": padding2,
        "out_type": out_type2,
        "dilations": dilations2,
        "name": name2,
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3
    input3 = np.array([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]], dtype=np.int8)
    filter3 = np.array([[[[1, 2, 3]], [[4, 5, 6]]]], dtype=np.int8)
    min_input3 = np.array([-128.0], dtype=np.float32)
    max_input3 = np.array([127.0], dtype=np.float32)
    min_filter3 = np.array([-128.0], dtype=np.float32)
    max_filter3 = np.array([127.0], dtype=np.float32)
    strides3 = [1, 1, 1, 1]
    padding3 = "VALID"
    out_type3 = tf.qint32
    dilations3 = [1, 1, 1, 1]
    name3 = "conv3"

    input_dict3 = {
        "input": tf.constant(input3, dtype=tf.qint8),
        "filter": tf.constant(filter3, dtype=tf.qint8),
        "min_input": tf.constant(min_input3, dtype=tf.float32),
        "max_input": tf.constant(max_input3, dtype=tf.float32),
        "min_filter": tf.constant(min_filter3, dtype=tf.float32),
        "max_filter": tf.constant(max_filter3, dtype=tf.float32),
        "strides": strides3,
        "padding": padding3,
        "out_type": out_type3,
        "dilations": dilations3,
        "name": name3,
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4
    input4 = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.uint8)
    filter4 = np.array([[[[1, 2]], [[3, 4]]]], dtype=np.uint8)
    min_input4 = np.array([0.0], dtype=np.float32)
    max_input4 = np.array([255.0], dtype=np.float32)
    min_filter4 = np.array([0.0], dtype=np.float32)
    max_filter4 = np.array([255.0], dtype=np.float32)
    strides4 = [1, 2, 2, 1]
    padding4 = "SAME"
    out_type4 = tf.qint32
    dilations4 = [1, 1, 1, 1]
    name4 = "conv4"

    input_dict4 = {
        "input": tf.constant(input4, dtype=tf.quint8),
        "filter": tf.constant(filter4, dtype=tf.quint8),
        "min_input": tf.constant(min_input4, dtype=tf.float32),
        "max_input": tf.constant(max_input4, dtype=tf.float32),
        "min_filter": tf.constant(min_filter4, dtype=tf.float32),
        "max_filter": tf.constant(max_filter4, dtype=tf.float32),
        "strides": strides4,
        "padding": padding4,
        "out_type": out_type4,
        "dilations": dilations4,
        "name": name4,
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5
    input5 = np.array([[[[1], [2]], [[3], [4]]]], dtype=np.int8)
    filter5 = np.array([[[[1]], [[2]]]], dtype=np.int8)
    min_input5 = np.array([-5.0], dtype=np.float32)
    max_input5 = np.array([5.0], dtype=np.float32)
    min_filter5 = np.array([-3.0], dtype=np.float32)
    max_filter5 = np.array([3.0], dtype=np.float32)
    strides5 = [1, 1, 1, 1]
    padding5 = "VALID"
    out_type5 = tf.qint32
    dilations5 = [1, 2, 2, 1]
    name5 = "conv5"

    input_dict5 = {
        "input": tf.constant(input5, dtype=tf.qint8),
        "filter": tf.constant(filter5, dtype=tf.qint8),
        "min_input": tf.constant(min_input5, dtype=tf.float32),
        "max_input": tf.constant(max_input5, dtype=tf.float32),
        "min_filter": tf.constant(min_filter5, dtype=tf.float32),
        "max_filter": tf.constant(max_filter5, dtype=tf.float32),
        "strides": strides5,
        "padding": padding5,
        "out_type": out_type5,
        "dilations": dilations5,
        "name": name5,
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6
    input6 = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.uint8)
    filter6 = np.array([[[[1, 2]], [[3, 4]]]], dtype=np.uint8)
    min_input6 = np.array([0.0], dtype=np.float32)
    max_input6 = np.array([255.0], dtype=np.float32)
    min_filter6 = np.array([0.0], dtype=np.float32)
    max_filter6 = np.array([255.0], dtype=np.float32)
    strides6 = [1, 1, 1, 1]
    padding6 = "SAME"
    out_type6 = tf.qint8
    dilations6 = [1, 1, 1, 1]
    name6 = "conv6"

    input_dict6 = {
        "input": tf.constant(input6, dtype=tf.quint8),
        "filter": tf.constant(filter6, dtype=tf.quint8),
        "min_input": tf.constant(min_input6, dtype=tf.float32),
        "max_input": tf.constant(max_input6, dtype=tf.float32),
        "min_filter": tf.constant(min_filter6, dtype=tf.float32),
        "max_filter": tf.constant(max_filter6, dtype=tf.float32),
        "strides": strides6,
        "padding": padding6,
        "out_type": out_type6,
        "dilations": dilations6,
        "name": name6,
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7
    input7 = np.array([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]], dtype=np.int16)
    filter7 = np.array([[[[1, 2, 3]], [[4, 5, 6]]]], dtype=np.int16)
    min_input7 = np.array([-32768.0], dtype=np.float32)
    max_input7 = np.array([32767.0], dtype=np.float32)
    min_filter7 = np.array([-32768.0], dtype=np.float32)
    max_filter7 = np.array([32767.0], dtype=np.float32)
    strides7 = [1, 1, 1, 1]
    padding7 = "VALID"
    out_type7 = tf.qint32
    dilations7 = [1, 1, 1, 1]
    name7 = "conv7"

    input_dict7 = {
        "input": tf.constant(input7, dtype=tf.qint16),
        "filter": tf.constant(filter7, dtype=tf.qint16),
        "min_input": tf.constant(min_input7, dtype=tf.float32),
        "max_input": tf.constant(max_input7, dtype=tf.float32),
        "min_filter": tf.constant(min_filter7, dtype=tf.float32),
        "max_filter": tf.constant(max_filter7, dtype=tf.float32),
        "strides": strides7,
        "padding": padding7,
        "out_type": out_type7,
        "dilations": dilations7,
        "name": name7,
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8
    input8 = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.uint16)
    filter8 = np.array([[[[1, 2]], [[3, 4]]]], dtype=np.uint16)
    min_input8 = np.array([0.0], dtype=np.float32)
    max_input8 = np.array([65535.0], dtype=np.float32)
    min_filter8 = np.array([0.0], dtype=np.float32)
    max_filter8 = np.array([65535.0], dtype=np.float32)
    strides8 = [1, 2, 2, 1]
    padding8 = "SAME"
    out_type8 = tf.qint32
    dilations8 = [1, 1, 1, 1]
    name8 = "conv8"

    input_dict8 = {
        "input": tf.constant(input8, dtype=tf.quint16),
        "filter": tf.constant(filter8, dtype=tf.quint16),
        "min_input": tf.constant(min_input8, dtype=tf.float32),
        "max_input": tf.constant(max_input8, dtype=tf.float32),
        "min_filter": tf.constant(min_filter8, dtype=tf.float32),
        "max_filter": tf.constant(max_filter8, dtype=tf.float32),
        "strides": strides8,
        "padding": padding8,
        "out_type": out_type8,
        "dilations": dilations8,
        "name": name8,
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9
    input9 = np.array([[[[1], [2]], [[3], [4]]]], dtype=np.int32)
    filter9 = np.array([[[[1]], [[2]]]], dtype=np.int32)
    min_input9 = np.array([-5.0], dtype=np.float32)
    max_input9 = np.array([5.0], dtype=np.float32)
    min_filter9 = np.array([-3.0], dtype=np.float32)
    max_filter9 = np.array([3.0], dtype=np.float32)
    strides9 = [1, 1, 1, 1]
    padding9 = "VALID"
    out_type9 = tf.qint32
    dilations9 = [1, 2, 2, 1]
    name9 = "conv9"

    input_dict9 = {
        "input": tf.constant(input9, dtype=tf.qint32),
        "filter": tf.constant(filter9, dtype=tf.qint32),
        "min_input": tf.constant(min_input9, dtype=tf.float32),
        "max_input": tf.constant(max_input9, dtype=tf.float32),
        "min_filter": tf.constant(min_filter9, dtype=tf.float32),
        "max_filter": tf.constant(max_filter9, dtype=tf.float32),
        "strides": strides9,
        "padding": padding9,
        "out_type": out_type9,
        "dilations": dilations9,
        "name": name9,
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10
    input10 = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.int16)
    filter10 = np.array([[[[1, 2]], [[3, 4]]]], dtype=np.int16)
    min_input10 = np.array([0.0], dtype=np.float32)
    max_input10 = np.array([255.0], dtype=np.float32)
    min_filter10 = np.array([0.0], dtype=np.float32)
    max_filter10 = np.array([255.0], dtype=np.float32)
    strides10 = [1, 1, 1, 1]
    padding10 = "SAME"
    out_type10 = tf.qint16
    dilations10 = [1, 1, 1, 1]
    name10 = "conv10"

    input_dict10 = {
        "input": tf.constant(input10, dtype=tf.qint16),
        "filter": tf.constant(filter10, dtype=tf.qint16),
        "min_input": tf.constant(min_input10, dtype=tf.float32),
        "max_input": tf.constant(max_input10, dtype=tf.float32),
        "min_filter": tf.constant(min_filter10, dtype=tf.float32),
        "max_filter": tf.constant(max_filter10, dtype=tf.float32),
        "strides": strides10,
        "padding": padding10,
        "out_type": out_type10,
        "dilations": dilations10,
        "name": name10,
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.QuantizedConv2D"] = tf_raw_ops_QuantizedConv2D_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.QuantizedConv2D' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QuantizedConv2D'.")

check_valid('tf.raw_ops.QuantizedConv2D', generated_inputs['tf.raw_ops.QuantizedConv2D'], lib="tf", suffix=0)
