
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def get_tf_raw_ops_quantized_relu6_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.QuantizedRelu6 function.
    The user's testing harness fails when presented with tf.Tensor objects
    that have quantized dtypes (e.g., tf.qint8), resulting in a ValueError.
    To resolve this specific error, all tensor inputs are provided strictly
    in numpy format, as requested by the prompt.
    """
    list_of_inputs = []

    # Input 1: quint8 case
    input_dict = {
        'features': np.array([0, 128, 153, 255], dtype=np.uint8),
        'min_features': np.array(-10.0, dtype=np.float32),
        'max_features': np.array(10.0, dtype=np.float32),
        'out_type': tf.quint8,
        'name': 'quint8_basic'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: qint8 case
    input_dict = {
        'features': np.array([-128, -25, 0, 100, 127], dtype=np.int8),
        'min_features': np.array(-8.0, dtype=np.float32),
        'max_features': np.array(8.0, dtype=np.float32),
        'out_type': tf.qint8,
        'name': 'qint8_basic'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D features with qint8
    input_dict = {
        'features': np.array([[-128, 0], [100, 127]], dtype=np.int8),
        'min_features': np.array(0.0, dtype=np.float32),
        'max_features': np.array(8.0, dtype=np.float32),
        'out_type': tf.qint8,
        'name': '2d_qint8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: qint16 type
    input_dict = {
        'features': np.array([-32768, 0, 10000, 32767], dtype=np.int16),
        'min_features': np.array(-30.0, dtype=np.float32),
        'max_features': np.array(30.0, dtype=np.float32),
        'out_type': tf.qint16,
        'name': 'qint16_test'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: quint16 type with 3D tensor
    input_dict = {
        'features': np.array([[[0, 1000], [32767, 40000]], [[50000, 60000], [65535, 20000]]], dtype=np.uint16),
        'min_features': np.array(0.0, dtype=np.float32),
        'max_features': np.array(12.0, dtype=np.float32),
        'out_type': tf.quint16,
        'name': 'quint16_3d_test'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: qint32 type
    input_dict = {
        'features': np.array([-2147483648, -10000, 0, 128849018, 2147483647], dtype=np.int32),
        'min_features': np.array(-100.0, dtype=np.float32),
        'max_features': np.array(100.0, dtype=np.float32),
        'out_type': tf.qint32,
        'name': 'qint32_test'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different out_type from features type (int32 to quint8)
    input_dict = {
        'features': np.array([-100, 0, 100, 200], dtype=np.int32),
        'min_features': np.array(-100.0, dtype=np.float32),
        'max_features': np.array(100.0, dtype=np.float32),
        'out_type': tf.quint8,
        'name': 'int32_to_quint8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Empty features tensor
    input_dict = {
        'features': np.array([], dtype=np.int8),
        'min_features': np.array(-1.0, dtype=np.float32),
        'max_features': np.array(1.0, dtype=np.float32),
        'out_type': tf.qint8,
        'name': 'empty_features'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: All values clipped below 0.0
    input_dict = {
        'features': np.array([0, 128, 255], dtype=np.uint8),
        'min_features': np.array(-10.0, dtype=np.float32),
        'max_features': np.array(-5.0, dtype=np.float32),
        'out_type': tf.quint8,
        'name': 'all_clipped_below'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Range is exactly [0, 6]
    input_dict = {
        'features': np.array([0, 128, 255], dtype=np.uint8),
        'min_features': np.array(0.0, dtype=np.float32),
        'max_features': np.array(6.0, dtype=np.float32),
        'out_type': tf.quint8,
        'name': 'range_0_to_6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Scalar features
    input_dict = {
        'features': np.array(100, dtype=np.int8),
        'min_features': np.array(-10.0, dtype=np.float32),
        'max_features': np.array(10.0, dtype=np.float32),
        'out_type': tf.qint8,
        'name': 'scalar_qint8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.QuantizedRelu6"] = get_tf_raw_ops_quantized_relu6_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.QuantizedRelu6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QuantizedRelu6'.")

check_valid('tf.raw_ops.QuantizedRelu6', generated_inputs['tf.raw_ops.QuantizedRelu6'], lib="tf", suffix=0)
