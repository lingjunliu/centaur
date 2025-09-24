
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def tf_raw_ops_QuantizedMatMul_inputs():
    list_of_inputs = []

    # This API requires 'a' and 'b' to be Tensors with specific quantized dtypes
    # (e.g., tf.qint8), which are created using tf.quantization.quantize.
    # Standard numpy arrays with integer dtypes (e.g., np.int8) are not valid
    # and will cause a TensorFlow InvalidArgumentError.
    def _get_quantized_tensors(shape, q_dtype_tf):
        """Generates a valid quantized tf.Tensor and its numpy min/max range."""
        float_data = np.random.uniform(low=-10.0, high=10.0, size=shape).astype(np.float32)
        min_val = np.min(float_data)
        max_val = np.max(float_data)
        if min_val >= max_val:
            max_val = min_val + 1.0
        q_tensor, q_min, q_max = tf.quantization.quantize(
            tf.constant(float_data), min_val, max_val, T=q_dtype_tf)
        # The signature requires min/max to be 'tensor', so we use 0-d numpy arrays.
        return q_tensor, np.array(q_min.numpy(), dtype=np.float32), np.array(q_max.numpy(), dtype=np.float32)

    # Input 1: Basic case with qint8
    a_q, min_a, max_a = _get_quantized_tensors((2, 3), tf.qint8)
    b_q, min_b, max_b = _get_quantized_tensors((3, 4), tf.qint8)
    input_dict = {
        'a': a_q, 'b': b_q, 'min_a': min_a, 'max_a': max_a, 'min_b': min_b, 'max_b': max_b,
        'Toutput': tf.qint32, 'transpose_a': False, 'transpose_b': False, 'Tactivation': tf.quint8, 'name': 'valid_1_qint8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: transpose_a=True with quint8
    a_q, min_a, max_a = _get_quantized_tensors((3, 2), tf.quint8)
    b_q, min_b, max_b = _get_quantized_tensors((3, 4), tf.quint8)
    input_dict = {
        'a': a_q, 'b': b_q, 'min_a': min_a, 'max_a': max_a, 'min_b': min_b, 'max_b': max_b,
        'Toutput': tf.qint32, 'transpose_a': True, 'transpose_b': False, 'Tactivation': tf.quint8, 'name': 'valid_2_transpose_a'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: transpose_b=True with qint16
    a_q, min_a, max_a = _get_quantized_tensors((2, 3), tf.qint16)
    b_q, min_b, max_b = _get_quantized_tensors((4, 3), tf.qint16)
    input_dict = {
        'a': a_q, 'b': b_q, 'min_a': min_a, 'max_a': max_a, 'min_b': min_b, 'max_b': max_b,
        'Toutput': tf.qint32, 'transpose_a': False, 'transpose_b': True, 'Tactivation': tf.qint16, 'name': 'valid_3_transpose_b'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Both transposes with quint16
    a_q, min_a, max_a = _get_quantized_tensors((3, 2), tf.quint16)
    b_q, min_b, max_b = _get_quantized_tensors((4, 3), tf.quint16)
    input_dict = {
        'a': a_q, 'b': b_q, 'min_a': min_a, 'max_a': max_a, 'min_b': min_b, 'max_b': max_b,
        'Toutput': tf.qint32, 'transpose_a': True, 'transpose_b': True, 'Tactivation': tf.quint16, 'name': 'valid_4_transpose_both'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different Toutput (qint8)
    a_q, min_a, max_a = _get_quantized_tensors((5, 5), tf.qint8)
    b_q, min_b, max_b = _get_quantized_tensors((5, 5), tf.qint8)
    input_dict = {
        'a': a_q, 'b': b_q, 'min_a': min_a, 'max_a': max_a, 'min_b': min_b, 'max_b': max_b,
        'Toutput': tf.qint8, 'transpose_a': False, 'transpose_b': False, 'Tactivation': tf.qint8, 'name': 'valid_5_toutput_qint8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: qint32 inputs
    a_q, min_a, max_a = _get_quantized_tensors((3, 3), tf.qint32)
    b_q, min_b, max_b = _get_quantized_tensors((3, 3), tf.qint32)
    input_dict = {
        'a': a_q, 'b': b_q, 'min_a': min_a, 'max_a': max_a, 'min_b': min_b, 'max_b': max_b,
        'Toutput': tf.qint32, 'transpose_a': False, 'transpose_b': False, 'Tactivation': tf.qint32, 'name': 'valid_6_qint32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large matrices
    a_q, min_a, max_a = _get_quantized_tensors((10, 20), tf.qint8)
    b_q, min_b, max_b = _get_quantized_tensors((20, 5), tf.qint8)
    input_dict = {
        'a': a_q, 'b': b_q, 'min_a': min_a, 'max_a': max_a, 'min_b': min_b, 'max_b': max_b,
        'Toutput': tf.qint32, 'transpose_a': False, 'transpose_b': False, 'Tactivation': tf.quint8, 'name': 'valid_7_large'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Vector-like matrices
    a_q, min_a, max_a = _get_quantized_tensors((1, 8), tf.quint8)
    b_q, min_b, max_b = _get_quantized_tensors((8, 1), tf.quint8)
    input_dict = {
        'a': a_q, 'b': b_q, 'min_a': min_a, 'max_a': max_a, 'min_b': min_b, 'max_b': max_b,
        'Toutput': tf.qint32, 'transpose_a': False, 'transpose_b': False, 'Tactivation': tf.quint8, 'name': 'valid_8_vector'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Toutput=quint16
    a_q, min_a, max_a = _get_quantized_tensors((6, 2), tf.quint8)
    b_q, min_b, max_b = _get_quantized_tensors((2, 7), tf.quint8)
    input_dict = {
        'a': a_q, 'b': b_q, 'min_a': min_a, 'max_a': max_a, 'min_b': min_b, 'max_b': max_b,
        'Toutput': tf.quint16, 'transpose_a': False, 'transpose_b': False, 'Tactivation': tf.quint16, 'name': 'valid_9_toutput_quint16'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Mismatched input types (qint8 and quint8)
    a_q, min_a, max_a = _get_quantized_tensors((4, 6), tf.qint8)
    b_q, min_b, max_b = _get_quantized_tensors((6, 2), tf.quint8)
    input_dict = {
        'a': a_q, 'b': b_q, 'min_a': min_a, 'max_a': max_a, 'min_b': min_b, 'max_b': max_b,
        'Toutput': tf.qint32, 'transpose_a': False, 'transpose_b': False, 'Tactivation': tf.quint8, 'name': 'valid_10_mismatched_types'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.QuantizedMatMul"] = tf_raw_ops_QuantizedMatMul_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.QuantizedMatMul' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QuantizedMatMul'.")

check_valid('tf.raw_ops.QuantizedMatMul', generated_inputs['tf.raw_ops.QuantizedMatMul'], lib="tf", suffix=0)
