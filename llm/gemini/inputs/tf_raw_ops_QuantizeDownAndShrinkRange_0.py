
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_QuantizeDownAndShrinkRange_inputs():
    list_of_inputs = []

    # Input 1: quint8 to quint8, minimal range
    input_tensor = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    input_min_tensor = np.array(0.0, dtype=np.float32)
    input_max_tensor = np.array(5.0, dtype=np.float32)
    out_type_val = tf.quint8

    input_dict = {
        "name": "quantize_down_and_shrink_range_1",
        "input": tf.constant(input_tensor, dtype=tf.quint8),
        "input_min": input_min_tensor,
        "input_max": input_max_tensor,
        "out_type": out_type_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: qint8 to quint8, negative range
    input_tensor = np.array([[-1, 0], [1, 2]], dtype=np.int8)
    input_min_tensor = np.array(-2.0, dtype=np.float32)
    input_max_tensor = np.array(3.0, dtype=np.float32)
    out_type_val = tf.quint8

    input_dict = {
        "name": "quantize_down_and_shrink_range_2",
        "input": tf.constant(input_tensor, dtype=tf.qint8),
        "input_min": input_min_tensor,
        "input_max": input_max_tensor,
        "out_type": out_type_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: quint16 to quint8, wider range
    input_tensor = np.array([[1000, 2000], [3000, 4000]], dtype=np.uint16)
    input_min_tensor = np.array(0.0, dtype=np.float32)
    input_max_tensor = np.array(5000.0, dtype=np.float32)
    out_type_val = tf.quint8

    input_dict = {
        "name": "quantize_down_and_shrink_range_3",
        "input": tf.constant(input_tensor, dtype=tf.quint16),
        "input_min": input_min_tensor,
        "input_max": input_max_tensor,
        "out_type": out_type_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: qint32 to qint8, negative range
    input_tensor = np.array([[-1000, 0], [1000, 2000]], dtype=np.int32)
    input_min_tensor = np.array(-2000.0, dtype=np.float32)
    input_max_tensor = np.array(3000.0, dtype=np.float32)
    out_type_val = tf.qint8

    input_dict = {
        "name": "quantize_down_and_shrink_range_4",
        "input": tf.constant(input_tensor, dtype=tf.qint32),
        "input_min": input_min_tensor,
        "input_max": input_max_tensor,
        "out_type": out_type_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 5: quint8 to qint8, zero range
    input_tensor = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    input_min_tensor = np.array(0.0, dtype=np.float32)
    input_max_tensor = np.array(0.0, dtype=np.float32)
    out_type_val = tf.qint8

    input_dict = {
        "name": "quantize_down_and_shrink_range_5",
        "input": tf.constant(input_tensor, dtype=tf.quint8),
        "input_min": input_min_tensor,
        "input_max": input_max_tensor,
        "out_type": out_type_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: qint16 to quint8, 3D tensor
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int16)
    input_min_tensor = np.array(-10.0, dtype=np.float32)
    input_max_tensor = np.array(10.0, dtype=np.float32)
    out_type_val = tf.quint8

    input_dict = {
        "name": "quantize_down_and_shrink_range_6",
        "input": tf.constant(input_tensor, dtype=tf.qint16),
        "input_min": input_min_tensor,
        "input_max": input_max_tensor,
        "out_type": out_type_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: quint8 to qint16
    input_tensor = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    input_min_tensor = np.array(0.0, dtype=np.float32)
    input_max_tensor = np.array(255.0, dtype=np.float32)
    out_type_val = tf.qint16

    input_dict = {
        "name": "quantize_down_and_shrink_range_7",
        "input": tf.constant(input_tensor, dtype=tf.quint8),
        "input_min": input_min_tensor,
        "input_max": input_max_tensor,
        "out_type": out_type_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: qint8 to qint16, 1D tensor
    input_tensor = np.array([-1, 0, 1, 2], dtype=np.int8)
    input_min_tensor = np.array(-128.0, dtype=np.float32)
    input_max_tensor = np.array(127.0, dtype=np.float32)
    out_type_val = tf.qint16

    input_dict = {
        "name": "quantize_down_and_shrink_range_8",
        "input": tf.constant(input_tensor, dtype=tf.qint8),
        "input_min": input_min_tensor,
        "input_max": input_max_tensor,
        "out_type": out_type_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: quint16 to qint8
    input_tensor = np.array([[1000, 2000], [3000, 4000]], dtype=np.uint16)
    input_min_tensor = np.array(0.0, dtype=np.float32)
    input_max_tensor = np.array(65535.0, dtype=np.float32)
    out_type_val = tf.qint8

    input_dict = {
        "name": "quantize_down_and_shrink_range_9",
        "input": tf.constant(input_tensor, dtype=tf.quint16),
        "input_min": input_min_tensor,
        "input_max": input_max_tensor,
        "out_type": out_type_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: qint32 to qint16, large range, multiple dimensions
    input_tensor = np.array([[[1, -2], [3, -4]], [[5, -6], [7, -8]]], dtype=np.int32)
    input_min_tensor = np.array(-100000.0, dtype=np.float32)
    input_max_tensor = np.array(100000.0, dtype=np.float32)
    out_type_val = tf.qint16

    input_dict = {
        "name": "quantize_down_and_shrink_range_10",
        "input": tf.constant(input_tensor, dtype=tf.qint32),
        "input_min": input_min_tensor,
        "input_max": input_max_tensor,
        "out_type": out_type_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.QuantizeDownAndShrinkRange"] = tf_raw_ops_QuantizeDownAndShrinkRange_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.QuantizeDownAndShrinkRange' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QuantizeDownAndShrinkRange'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.QuantizeDownAndShrinkRange', generated_inputs['tf.raw_ops.QuantizeDownAndShrinkRange'], lib="tf", suffix=0)
