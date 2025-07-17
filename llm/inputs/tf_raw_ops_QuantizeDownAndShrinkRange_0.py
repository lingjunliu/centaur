
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_QuantizeDownAndShrinkRange_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.qint16)
    input_min_tensor = np.array(-1.0, dtype=np.float32)
    input_max_tensor = np.array(1.0, dtype=np.float32)
    out_type = tf.quint8

    input_dict = {
        "name": "quantize_down_and_shrink_range_1",
        "input": input_tensor,
        "input_min": input_min_tensor,
        "input_max": input_max_tensor,
        "out_type": out_type
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[-1, 0, 1], [2, 3, 4]], dtype=np.qint8)
    input_min_tensor = np.array(-5.0, dtype=np.float32)
    input_max_tensor = np.array(5.0, dtype=np.float32)
    out_type = tf.quint8

    input_dict = {
        "name": "quantize_down_and_shrink_range_2",
        "input": input_tensor,
        "input_min": input_min_tensor,
        "input_max": input_max_tensor,
        "out_type": out_type
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([1000, 2000, 3000, 4000], dtype=np.qint32)
    input_min_tensor = np.array(0.0, dtype=np.float32)
    input_max_tensor = np.array(5000.0, dtype=np.float32)
    out_type = tf.quint8

    input_dict = {
        "name": "quantize_down_and_shrink_range_3",
        "input": input_tensor,
        "input_min": input_min_tensor,
        "input_max": input_max_tensor,
        "out_type": out_type
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([[1, 2], [3, 4]], dtype=np.quint16)
    input_min_tensor = np.array(0.0, dtype=np.float32)
    input_max_tensor = np.array(10.0, dtype=np.float32)
    out_type = tf.quint8

    input_dict = {
        "name": "quantize_down_and_shrink_range_4",
        "input": input_tensor,
        "input_min": input_min_tensor,
        "input_max": input_max_tensor,
        "out_type": out_type
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([[-10, -5, 0, 5, 10]], dtype=np.qint16)
    input_min_tensor = np.array(-15.0, dtype=np.float32)
    input_max_tensor = np.array(15.0, dtype=np.float32)
    out_type = tf.quint8

    input_dict = {
        "name": "quantize_down_and_shrink_range_5",
        "input": input_tensor,
        "input_min": input_min_tensor,
        "input_max": input_max_tensor,
        "out_type": out_type
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array([1, 2, 3, 4, 5], dtype=np.quint8)
    input_min_tensor = np.array(0.0, dtype=np.float32)
    input_max_tensor = np.array(255.0, dtype=np.float32)
    out_type = tf.quint8

    input_dict = {
        "name": "quantize_down_and_shrink_range_6",
        "input": input_tensor,
        "input_min": input_min_tensor,
        "input_max": input_max_tensor,
        "out_type": out_type
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array([[-1, -2], [-3, -4]], dtype=np.qint16)
    input_min_tensor = np.array(-5.0, dtype=np.float32)
    input_max_tensor = np.array(0.0, dtype=np.float32)
    out_type = tf.quint8

    input_dict = {
        "name": "quantize_down_and_shrink_range_7",
        "input": input_tensor,
        "input_min": input_min_tensor,
        "input_max": input_max_tensor,
        "out_type": out_type
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array([1, 2, 3, 4, 5], dtype=np.qint32)
    input_min_tensor = np.array(0.0, dtype=np.float32)
    input_max_tensor = np.array(10.0, dtype=np.float32)
    out_type = tf.quint8

    input_dict = {
        "name": "quantize_down_and_shrink_range_8",
        "input": input_tensor,
        "input_min": input_min_tensor,
        "input_max": input_max_tensor,
        "out_type": out_type
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 9
    input_tensor = np.array([1, 2, 3, 4, 5], dtype=np.quint16)
    input_min_tensor = np.array(0.0, dtype=np.float32)
    input_max_tensor = np.array(65535.0, dtype=np.float32)
    out_type = tf.quint8

    input_dict = {
        "name": "quantize_down_and_shrink_range_9",
        "input": input_tensor,
        "input_min": input_min_tensor,
        "input_max": input_max_tensor,
        "out_type": out_type
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.array([1, 2, 3, 4, 5], dtype=np.qint8)
    input_min_tensor = np.array(0.0, dtype=np.float32)
    input_max_tensor = np.array(255.0, dtype=np.float32)
    out_type = tf.quint8

    input_dict = {
        "name": "quantize_down_and_shrink_range_10",
        "input": input_tensor,
        "input_min": input_min_tensor,
        "input_max": input_max_tensor,
        "out_type": out_type
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.QuantizeDownAndShrinkRange"] = tf_raw_ops_QuantizeDownAndShrinkRange_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.QuantizeDownAndShrinkRange' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QuantizeDownAndShrinkRange'.")

check_valid('tf.raw_ops.QuantizeDownAndShrinkRange', generated_inputs['tf.raw_ops.QuantizeDownAndShrinkRange'], lib="tf", suffix=0)
