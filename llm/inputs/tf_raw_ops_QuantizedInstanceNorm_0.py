
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def get_quantized_instance_norm_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.QuantizedInstanceNorm function.
    The 'x' tensor is created as a tf.Tensor with a quantized dtype (e.g., tf.qint8),
    as this is strictly required by the API.
    """
    list_of_inputs = []

    def quantize_input(float_data, min_range, max_range, q_type, shape):
        """Helper to create quantized tensors and scalar float tensors for min/max."""
        x_float = tf.constant(float_data, dtype=tf.float32, shape=shape)
        # mode="SCALED" is used for quantization
        q, q_min, q_max = tf.quantization.quantize(x_float, min_range, max_range, T=q_type, mode="SCALED")
        # The op expects scalar tensors for min/max.
        return q, tf.constant(q_min, dtype=tf.float32), tf.constant(q_max, dtype=tf.float32)

    # Input 1: Basic case with qint8, auto output range
    q_x, q_min, q_max = quantize_input(np.arange(-12, 12, 1.0), -12.0, 11.0, tf.qint8, (1, 2, 3, 4))
    input_dict_1 = {
        'x': q_x,
        'x_min': q_min,
        'x_max': q_max,
        'output_range_given': False,
        'given_y_min': 0.0,
        'given_y_max': 0.0,
        'variance_epsilon': 1e-05,
        'min_separation': 0.001,
        'name': 'test_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Basic case with quint8, auto output range
    q_x, q_min, q_max = quantize_input(np.arange(0, 18, 1.0), 0.0, 17.0, tf.quint8, (2, 3, 3, 1))
    input_dict_2 = {
        'x': q_x,
        'x_min': q_min,
        'x_max': q_max,
        'output_range_given': False,
        'given_y_min': 0.0,
        'given_y_max': 0.0,
        'variance_epsilon': 1e-05,
        'min_separation': 0.001,
        'name': 'test_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: qint16 with given output range
    q_x, q_min, q_max = quantize_input(np.random.uniform(-1000, 1000, size=50), -1000.0, 1000.0, tf.qint16, (1, 5, 5, 2))
    input_dict_3 = {
        'x': q_x,
        'x_min': q_min,
        'x_max': q_max,
        'output_range_given': True,
        'given_y_min': -1.0,
        'given_y_max': 1.0,
        'variance_epsilon': 1e-05,
        'min_separation': 0.001,
        'name': 'test_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: quint16 with given output range and custom epsilon/separation
    q_x, q_min, q_max = quantize_input(np.random.uniform(100, 5000, size=24), 100.0, 5000.0, tf.quint16, (3, 2, 2, 2))
    input_dict_4 = {
        'x': q_x,
        'x_min': q_min,
        'x_max': q_max,
        'output_range_given': True,
        'given_y_min': 0.0,
        'given_y_max': 10.0,
        'variance_epsilon': 1e-04,
        'min_separation': 0.01,
        'name': 'test_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: qint32 with auto output range
    q_x, q_min, q_max = quantize_input(np.random.uniform(-200000, 200000, size=16), -200000.0, 200000.0, tf.qint32, (2, 2, 2, 2))
    input_dict_5 = {
        'x': q_x,
        'x_min': q_min,
        'x_max': q_max,
        'output_range_given': False,
        'given_y_min': 0.0,
        'given_y_max': 0.0,
        'variance_epsilon': 1e-05,
        'min_separation': 0.001,
        'name': 'test_5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: qint8 with negative given output range
    q_x, q_min, q_max = quantize_input(np.random.uniform(-128.0, 0.0, size=16), -128.0, -1.0, tf.qint8, (1, 2, 2, 4))
    input_dict_6 = {
        'x': q_x,
        'x_min': q_min,
        'x_max': q_max,
        'output_range_given': True,
        'given_y_min': -5.0,
        'given_y_max': -0.5,
        'variance_epsilon': 1e-05,
        'min_separation': 0.001,
        'name': 'test_6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Uniform input (zeros), tests variance_epsilon
    q_x, q_min, q_max = quantize_input(np.zeros(1200), -1.0, 1.0, tf.qint8, (4, 10, 10, 3))
    input_dict_7 = {
        'x': q_x,
        'x_min': q_min,
        'x_max': q_max,
        'output_range_given': False,
        'given_y_min': 0.0,
        'given_y_max': 0.0,
        'variance_epsilon': 1e-02,
        'min_separation': 0.001,
        'name': 'test_7_zeros'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))
    
    # Input 8: x_max close to x_min, testing min_separation
    q_x, q_min, q_max = quantize_input(np.random.uniform(0.0, 0.0001, size=24), 0.0, 0.0001, tf.qint8, (1, 2, 3, 4))
    input_dict_8 = {
        'x': q_x,
        'x_min': q_min,
        'x_max': q_max,
        'output_range_given': False,
        'given_y_min': 0.0,
        'given_y_max': 0.0,
        'variance_epsilon': 1e-05,
        'min_separation': 0.1,
        'name': 'test_8_min_sep'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))
    
    # Input 9: Large range for quint16 with given output range
    q_x, q_min, q_max = quantize_input(np.arange(0, 48, 1.0), 0.0, 65535.0, tf.quint16, (1, 4, 4, 3))
    input_dict_9 = {
        'x': q_x,
        'x_min': q_min,
        'x_max': q_max,
        'output_range_given': True,
        'given_y_min': -100.0,
        'given_y_max': 100.0,
        'variance_epsilon': 1e-05,
        'min_separation': 0.001,
        'name': 'test_9_large_range'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))
    
    # Input 10: Another qint32 example
    q_x, q_min, q_max = quantize_input(np.random.uniform(-500, 500, size=18), -500.0, 500.0, tf.qint32, (1, 3, 3, 2))
    input_dict_10 = {
        'x': q_x,
        'x_min': q_min,
        'x_max': q_max,
        'output_range_given': True,
        'given_y_min': -1.0,
        'given_y_max': 1.0,
        'variance_epsilon': 1e-06,
        'min_separation': 0.0001,
        'name': 'test_10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.QuantizedInstanceNorm"] = get_quantized_instance_norm_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.QuantizedInstanceNorm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QuantizedInstanceNorm'.")

check_valid('tf.raw_ops.QuantizedInstanceNorm', generated_inputs['tf.raw_ops.QuantizedInstanceNorm'], lib="tf", suffix=0)
