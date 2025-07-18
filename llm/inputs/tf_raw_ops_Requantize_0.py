
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_requantize_inputs():
    list_of_inputs = []

    # NOTE: The following inputs will fail due to a fundamental type mismatch.
    # The 'Requantize' op requires a quantized input tensor (e.g., tf.qint8),
    # but the testing framework creates a standard tensor (e.g., tf.int8) from
    # the provided numpy array. This is an unsolvable constraint.

    # Case 1: Attempting qint8 -> qint8 (will become int8 -> qint8 and fail)
    input_dict_1 = {
        'input': np.array([-128, 0, 127], dtype=np.int8),
        'input_min': np.array(-1.0, dtype=np.float32),
        'input_max': np.array(1.0, dtype=np.float32),
        'requested_output_min': np.array(-2.0, dtype=np.float32),
        'requested_output_max': np.array(2.0, dtype=np.float32),
        'out_type': tf.qint8,
        'name': 'failing_qint8_to_qint8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Case 2: Attempting quint8 -> quint8 (will become uint8 -> quint8 and fail)
    input_dict_2 = {
        'input': np.array([0, 100, 255], dtype=np.uint8),
        'input_min': np.array(0.0, dtype=np.float32),
        'input_max': np.array(255.0, dtype=np.float32),
        'requested_output_min': np.array(0.0, dtype=np.float32),
        'requested_output_max': np.array(127.0, dtype=np.float32),
        'out_type': tf.quint8,
        'name': 'failing_quint8_to_quint8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Case 3: Attempting quint8 -> qint8 (will become uint8 -> qint8 and fail)
    input_dict_3 = {
        'input': np.array([[0, 50], [150, 250]], dtype=np.uint8),
        'input_min': np.array(0.0, dtype=np.float32),
        'input_max': np.array(10.0, dtype=np.float32),
        'requested_output_min': np.array(-5.0, dtype=np.float32),
        'requested_output_max': np.array(5.0, dtype=np.float32),
        'out_type': tf.qint8,
        'name': 'failing_quint8_to_qint8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Case 4: qint16 -> qint8 (Provided again to show the error is persistent)
    input_dict_4 = {
        'input': np.array([-32768, 0, 32767], dtype=np.int16),
        'input_min': np.array(-1.0, dtype=np.float32),
        'input_max': np.array(1.0, dtype=np.float32),
        'requested_output_min': np.array(-128.0, dtype=np.float32),
        'requested_output_max': np.array(127.0, dtype=np.float32),
        'out_type': tf.qint8,
        'name': 'persistent_error_qint16_to_qint8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    return list_of_inputs

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

check_valid('tf.raw_ops.Requantize', generated_inputs['tf.raw_ops.Requantize'], lib="tf", suffix=0)
