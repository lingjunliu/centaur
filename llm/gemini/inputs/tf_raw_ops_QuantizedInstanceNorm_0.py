
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_QuantizedInstanceNorm_inputs():
    list_of_inputs = []

    # Input 1: Basic example with quint8
    x = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.uint8)
    x_min = np.array([0.0], dtype=np.float32)
    x_max = np.array([255.0], dtype=np.float32)
    x = tf.convert_to_tensor(x, dtype=tf.quint8)
    x_min = tf.convert_to_tensor(x_min, dtype=tf.float32)
    x_max = tf.convert_to_tensor(x_max, dtype=tf.float32)

    input_dict = {
        "x": x.numpy(),
        "x_min": x_min.numpy(),
        "x_max": x_max.numpy(),
        "output_range_given": False,
        "given_y_min": 0.0,
        "given_y_max": 0.0,
        "variance_epsilon": 1e-05,
        "min_separation": 0.001,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different data, qint8
    x = np.array([[[[-1, -2], [-3, -4]], [[-5, -6], [-7, -8]]]], dtype=np.int8)
    x_min = np.array([-128.0], dtype=np.float32)
    x_max = np.array([127.0], dtype=np.float32)
    x = tf.convert_to_tensor(x, dtype=tf.qint8)
    x_min = tf.convert_to_tensor(x_min, dtype=tf.float32)
    x_max = tf.convert_to_tensor(x_max, dtype=tf.float32)
    input_dict = {
        "x": x.numpy(),
        "x_min": x_min.numpy(),
        "x_max": x_max.numpy(),
        "output_range_given": False,
        "given_y_min": 0.0,
        "given_y_max": 0.0,
        "variance_epsilon": 1e-05,
        "min_separation": 0.001,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: output_range_given is True
    x = np.array([[[[10, 20], [30, 40]], [[50, 60], [70, 80]]]], dtype=np.uint8)
    x_min = np.array([0.0], dtype=np.float32)
    x_max = np.array([100.0], dtype=np.float32)

    x = tf.convert_to_tensor(x, dtype=tf.quint8)
    x_min = tf.convert_to_tensor(x_min, dtype=tf.float32)
    x_max = tf.convert_to_tensor(x_max, dtype=tf.float32)
    input_dict = {
        "x": x.numpy(),
        "x_min": x_min.numpy(),
        "x_max": x_max.numpy(),
        "output_range_given": True,
        "given_y_min": 10.0,
        "given_y_max": 50.0,
        "variance_epsilon": 1e-05,
        "min_separation": 0.001,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different variance_epsilon
    x = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.uint8)
    x_min = np.array([0.0], dtype=np.float32)
    x_max = np.array([255.0], dtype=np.float32)
    x = tf.convert_to_tensor(x, dtype=tf.quint8)
    x_min = tf.convert_to_tensor(x_min, dtype=tf.float32)
    x_max = tf.convert_to_tensor(x_max, dtype=tf.float32)
    input_dict = {
        "x": x.numpy(),
        "x_min": x_min.numpy(),
        "x_max": x_max.numpy(),
        "output_range_given": False,
        "given_y_min": 0.0,
        "given_y_max": 0.0,
        "variance_epsilon": 1e-03,
        "min_separation": 0.001,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different min_separation
    x = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.uint8)
    x_min = np.array([0.0], dtype=np.float32)
    x_max = np.array([255.0], dtype=np.float32)
    x = tf.convert_to_tensor(x, dtype=tf.quint8)
    x_min = tf.convert_to_tensor(x_min, dtype=tf.float32)
    x_max = tf.convert_to_tensor(x_max, dtype=tf.float32)
    input_dict = {
        "x": x.numpy(),
        "x_min": x_min.numpy(),
        "x_max": x_max.numpy(),
        "output_range_given": False,
        "given_y_min": 0.0,
        "given_y_max": 0.0,
        "variance_epsilon": 1e-05,
        "min_separation": 0.1,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: qint32
    x = np.array([[[[1000, 2000], [3000, 4000]], [[5000, 6000], [7000, 8000]]]], dtype=np.int32)
    x_min = np.array([-10000.0], dtype=np.float32)
    x_max = np.array([10000.0], dtype=np.float32)
    x = tf.convert_to_tensor(x, dtype=tf.qint32)
    x_min = tf.convert_to_tensor(x_min, dtype=tf.float32)
    x_max = tf.convert_to_tensor(x_max, dtype=tf.float32)
    input_dict = {
        "x": x.numpy(),
        "x_min": x_min.numpy(),
        "x_max": x_max.numpy(),
        "output_range_given": False,
        "given_y_min": 0.0,
        "given_y_max": 0.0,
        "variance_epsilon": 1e-05,
        "min_separation": 0.001,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: quint16
    x = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.uint16)
    x_min = np.array([0.0], dtype=np.float32)
    x_max = np.array([65535.0], dtype=np.float32)
    x = tf.convert_to_tensor(x, dtype=tf.quint16)
    x_min = tf.convert_to_tensor(x_min, dtype=tf.float32)
    x_max = tf.convert_to_tensor(x_max, dtype=tf.float32)
    input_dict = {
        "x": x.numpy(),
        "x_min": x_min.numpy(),
        "x_max": x_max.numpy(),
        "output_range_given": False,
        "given_y_min": 0.0,
        "given_y_max": 0.0,
        "variance_epsilon": 1e-05,
        "min_separation": 0.001,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: qint16
    x = np.array([[[[-1, -2], [-3, -4]], [[-5, -6], [-7, -8]]]], dtype=np.int16)
    x_min = np.array([-32768.0], dtype=np.float32)
    x_max = np.array([32767.0], dtype=np.float32)
    x = tf.convert_to_tensor(x, dtype=tf.qint16)
    x_min = tf.convert_to_tensor(x_min, dtype=tf.float32)
    x_max = tf.convert_to_tensor(x_max, dtype=tf.float32)
    input_dict = {
        "x": x.numpy(),
        "x_min": x_min.numpy(),
        "x_max": x_max.numpy(),
        "output_range_given": False,
        "given_y_min": 0.0,
        "given_y_max": 0.0,
        "variance_epsilon": 1e-05,
        "min_separation": 0.001,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: More complex data
    x = np.array([[[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], [[[13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24]]]]], dtype=np.uint8)
    x_min = np.array([0.0], dtype=np.float32)
    x_max = np.array([255.0], dtype=np.float32)
    x = tf.convert_to_tensor(x, dtype=tf.quint8)
    x_min = tf.convert_to_tensor(x_min, dtype=tf.float32)
    x_max = tf.convert_to_tensor(x_max, dtype=tf.float32)
    input_dict = {
        "x": x.numpy(),
        "x_min": x_min.numpy(),
        "x_max": x_max.numpy(),
        "output_range_given": False,
        "given_y_min": 0.0,
        "given_y_max": 0.0,
        "variance_epsilon": 1e-05,
        "min_separation": 0.001,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Different Shape
    x = np.array([[[[1, 2], [3, 4]]]], dtype=np.uint8)
    x_min = np.array([0.0], dtype=np.float32)
    x_max = np.array([255.0], dtype=np.float32)
    x = tf.convert_to_tensor(x, dtype=tf.quint8)
    x_min = tf.convert_to_tensor(x_min, dtype=tf.float32)
    x_max = tf.convert_to_tensor(x_max, dtype=tf.float32)
    input_dict = {
        "x": x.numpy(),
        "x_min": x_min.numpy(),
        "x_max": x_max.numpy(),
        "output_range_given": False,
        "given_y_min": 0.0,
        "given_y_max": 0.0,
        "variance_epsilon": 1e-05,
        "min_separation": 0.001,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.QuantizedInstanceNorm"] = tf_raw_ops_QuantizedInstanceNorm_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.QuantizedInstanceNorm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QuantizedInstanceNorm'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.QuantizedInstanceNorm', generated_inputs['tf.raw_ops.QuantizedInstanceNorm'], lib="tf", suffix=0)
