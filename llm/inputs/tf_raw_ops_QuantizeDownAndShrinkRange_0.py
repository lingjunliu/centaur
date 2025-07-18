
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import tensorflow as tf

def tf_raw_ops_quantize_down_and_shrink_range_inputs():
    """
    Generates a list of inputs for tf.raw_ops.QuantizeDownAndShrinkRange.
    The inputs use standard numpy dtypes (int32, int8, uint8) as the underlying
    representation for the required quantized types (qint32, qint8, quint8).
    This is a workaround for the test environment's inability to create true
    quantized tensors, which is the root cause of the Tinput error.
    """
    list_of_inputs = []

    # Input 1: Conceptual qint32 -> qint8.
    # Using np.int32 as the base type for the input tensor.
    input_dict = {
        'name': 'conceptual_qint32_to_qint8',
        'input': np.array([-100000, 0, 50000, 200000], dtype=np.int32),
        'input_min': np.array(-1.0, dtype=np.float32),
        'input_max': np.array(1.0, dtype=np.float32),
        'out_type': tf.qint8
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Conceptual qint32 -> quint16.
    input_dict = {
        'name': 'conceptual_qint32_to_quint16',
        'input': np.array([[10000, 20000], [30000, 40000]], dtype=np.int32),
        'input_min': np.array(0.0, dtype=np.float32),
        'input_max': np.array(100.0, dtype=np.float32),
        'out_type': tf.quint16
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Conceptual qint8 -> qint8 (shrinking range).
    # Using np.int8 as the base type for the input tensor.
    input_dict = {
        'name': 'conceptual_qint8_to_qint8_shrink',
        'input': np.array([-50, 0, 50], dtype=np.int8),
        'input_min': np.array(-128.0, dtype=np.float32),
        'input_max': np.array(127.0, dtype=np.float32),
        'out_type': tf.qint8
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Conceptual quint8 -> quint8 (shrinking range).
    # Using np.uint8 as the base type for the input tensor.
    input_dict = {
        'name': 'conceptual_quint8_to_quint8_shrink',
        'input': np.array([10, 20, 30], dtype=np.uint8),
        'input_min': np.array(0.0, dtype=np.float32),
        'input_max': np.array(255.0, dtype=np.float32),
        'out_type': tf.quint8
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Single element tensor.
    input_dict = {
        'name': 'single_element_qint32_to_qint8',
        'input': np.array([12345678], dtype=np.int32),
        'input_min': np.array(0.0, dtype=np.float32),
        'input_max': np.array(1.0, dtype=np.float32),
        'out_type': tf.qint8
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.QuantizeDownAndShrinkRange"] = tf_raw_ops_quantize_down_and_shrink_range_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.QuantizeDownAndShrinkRange' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QuantizeDownAndShrinkRange'.")

check_valid('tf.raw_ops.QuantizeDownAndShrinkRange', generated_inputs['tf.raw_ops.QuantizeDownAndShrinkRange'], lib="tf", suffix=0)
